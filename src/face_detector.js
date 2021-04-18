let cv = require('opencv')
module.exports = class FaceDetector {
    static detect = (path) => {
        const COLOR = [0, 255, 0];
        cv.readImage(path, function (err, im) {
            if (err) throw err;

            if (im.width() < 1 || im.height() < 1) throw new Error('Image has no size');
            im.detectObject("./src/data/opencv/haarcascade_frontalface_alt.xml", {}, function (err, faces) {
                if (err) throw err;
                for (var i = 0; i < faces.length; i++) {
                    var face = faces[i];
                    im.rectangle([face.x, face.y], [face.width, face.height], COLOR, 2);
                }
                im.save('./src/tmp/face-detection.png');
                console.log('Image saved to ./tmp/face-detection.png');
            });
        });
    }
}