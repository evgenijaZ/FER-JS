const express = require('express');
const lstmPredictor = require('../src/predictor')
const path = require("path");
const router = express.Router();
const bodyParser = require('body-parser');
const parser = bodyParser.json();

router.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '../public', 'index.html'));
});

lstmPredictor.create().then(predictor => {
        router.post('/', parser, function (req, res) {
            try {
                const run = async () => {
                    return await predictor.classifyFromArrays(req.body.images);
                }
                run().then(prediction => res.json({result: prediction}));
            } catch (e) {
                console.log(e)
                res.status(500).send()
            }
        });
    }
);

module.exports = router;