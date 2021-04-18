const express = require('express');
const resNetPredictor = require('../src/predictor')
const router = express.Router();

router.get('/', (req, res) => {
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