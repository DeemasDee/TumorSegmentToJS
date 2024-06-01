const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const { createCanvas, loadImage } = require('canvas');

// Load your model and labels
const labels = ["glioma", "notumor", "meningioma", "pituitary"];
let model;

const loadModel = async () => {
    model = await tf.loadLayersModel('file://tfjs_model/model.json');
};
loadModel();

const app = express();
const port = 3000;

const imageFolder = path.join(__dirname, 'static/image');

if (!fs.existsSync(imageFolder)) {
    fs.mkdirSync(imageFolder, { recursive: true });
}

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, imageFolder);
    },
    filename: (req, file, cb) => {
        cb(null, file.originalname);
    }
});

const upload = multer({ storage });

const preprocessImage = async (imagePath, imageSize) => {
    const img = await loadImage(imagePath);
    const canvas = createCanvas(imageSize, imageSize);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, imageSize, imageSize);
    const imageData = ctx.getImageData(0, 0, imageSize, imageSize);
    let data = tf.browser.fromPixels(imageData).toFloat().div(tf.scalar(255));
    data = data.expandDims(0);
    return data;
};

const predictTumorClass = async (model, imagePath, labels) => {
    const image = await preprocessImage(imagePath, 299);
    const prediction = model.predict(image);
    const predictionArray = await prediction.array();
    const predictedClass = labels[predictionArray[0].indexOf(Math.max(...predictionArray[0]))];
    const confidence = Math.max(...predictionArray[0]) * 100;
    return { predictedClass, confidence };
};

app.use(express.static(path.join(__dirname, 'static')));

app.post('/upload', upload.single('file'), async (req, res) => {
    const file = req.file;
    const inputFilePath = path.join(imageFolder, file.filename);

    try {
        const { predictedClass, confidence } = await predictTumorClass(model, inputFilePath, labels);
        res.json({ file: file.filename, predictedClass, confidence });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
