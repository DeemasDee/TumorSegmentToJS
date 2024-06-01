const express = require('express');
const multer = require('multer');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

let model;

// Load the model
(async () => {
    try {
        model = await tf.loadLayersModel('file://tfjs_model/model.json');
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Failed to load model:', error);
    }
})();

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));

// Handle file upload
app.post('/upload', upload.single('file'), async (req, res) => {
    try {
        console.log('File uploaded:', req.file);

        if (!model) {
            throw new Error('Model is not loaded');
        }

        const imagePath = req.file.path;
        const imageBuffer = fs.readFileSync(imagePath);
        console.log('Image buffer read successfully');

        const imageTensor = tf.node.decodeImage(imageBuffer, 3)
            .resizeNearestNeighbor([224, 224]) // adjust to the input shape of your model
            .expandDims()
            .toFloat()
            .div(tf.scalar(127))
            .sub(tf.scalar(1)); // normalization
        console.log('Image tensor created successfully');

        const predictions = model.predict(imageTensor);
        console.log('Prediction made successfully');

        const predictedClass = predictions.argMax(-1).dataSync()[0];
        const confidence = predictions.max().dataSync()[0] * 100;
        console.log(`Predicted Class: ${predictedClass}, Confidence: ${confidence}`);

        // Delete the uploaded file
        fs.unlinkSync(imagePath);
        console.log('Uploaded file deleted successfully');

        res.json({ predictedClass, confidence });
    } catch (error) {
        console.error('Error during prediction:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
