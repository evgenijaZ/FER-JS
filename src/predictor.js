var tf = require('@tensorflow/tfjs-node');
var Jimp = require('jimp');
var labels = require('./data/assets/labels.json');
var math = require('mathjs')
module.exports = class LSTMPredictor {
    constructor() {
        this.model;
        this.labels = labels;
        this.modelPath = `file://${__dirname}/data/model/20210425-013926_180_lstm_model_tf/model.json`;
    }

    initialize = async () => {
        this.model = await tf.loadLayersModel(this.modelPath);
    };

    static create = async () => {
        const o = new LSTMPredictor();
        await o.initialize();
        return o;
    };

    loadImg = async imgURI => {
        return Jimp.read(imgURI).then(img => {
            img.resize(48, 48);
            const p = [];
            img.scan(0, 0, img.bitmap.width, img.bitmap.height, function test(
                x,
                y,
                idx
            ) {
                p.push(0.2126 * this.bitmap.data[idx + 0] + 0.7152 * this.bitmap.data[idx + 1] + 0.0722 * this.bitmap.data[idx + 2]);
            });
            return p;
        });
    };

    classify = async imgURIs => {
        let values = []
        for (const file of imgURIs) {
            const contents = await this.loadImg(file);
            values.push(contents)
        }
        return await this.classifyFromArrays(values)
    };

    classifyFromArrays = async arrays => {
        let normalized = []
        arrays.forEach((arr) => {
                normalized.push(arr.map(x => x / 255))
            }
        )
        let shape = [normalized.length, 48, 48, 1];
        let reshaped = math.reshape(normalized, shape);
        const img = tf.tensor4d(reshaped, shape);
        let expanded = img.expandDims(0);
        const predictions = await this.model.predict(expanded);
        const predictionValues = predictions
            .reshape([5])
            .dataSync();
        let result1 = []
        for (let key in this.labels) {
            result1.push({emotion:labels[key], value:predictionValues[key]})
        }
        return result1;
    }
}