const express = require('express');
const resNetPredictor = require('../src/predictor')
const faceDetector = require('../src/face_detector')
const path = require("path");
const router = express.Router();

router.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public', 'index.html'));
});

router.post('/', function (req, res) {
    const prepare_faces = async () => {
        let coordinates = []
        coordinates.push(await faceDetector.detect("./src/data/images/img.png"));
        coordinates.push(await faceDetector.detect("./src/data/images/img_1.png"));
        coordinates.push(await faceDetector.detect("./src/data/images/img_2.png"));
        return coordinates;
    }
    prepare_faces().then(coordinates => console.log("index.js:" + coordinates + "."))

    const IMG1 = './src/data/images/S010_006_00000013.png'
    const IMG2 = './src/data/images/S010_006_00000014.png'
    const IMG3 = './src/data/images/S010_006_00000015.png'
    const run = async () => {
        const predictor = await resNetPredictor.create();
        return await predictor.classify([IMG1, IMG2, IMG3]);
    }
    run().then(prediction => res.send(prediction));
});

module.exports = router;