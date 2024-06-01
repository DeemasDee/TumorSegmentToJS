const tf = require('@tensorflow/tfjs-node');

(async () => {
    try {
        const model = await tf.loadLayersModel('file://tfjs_model/model.json');
        console.log('Model loaded successfully');
        // Optionally, make a dummy prediction to ensure everything is set up
        const dummyInput = tf.zeros([1, 224, 224, 3]);
        const dummyOutput = model.predict(dummyInput);
        dummyOutput.print();
    } catch (error) {
        console.error('Failed to load model:', error);
    }
})();
