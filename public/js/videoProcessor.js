function VideoProcessor(errorOutputId) { // eslint-disable-line no-unused-vars
    let self = this;
    this.errorOutput = document.getElementById(errorOutputId);

    const OPENCV_URL = 'js/opencv.js';
    this.loadOpenCv = onloadCallback => {
        let script = document.createElement('script');
        script.setAttribute('async', '');
        script.setAttribute('type', 'text/javascript');
        script.addEventListener('load', async () => {
            if (cv.getBuildInformation) {
                console.log(cv.getBuildInformation());
                onloadCallback();
            } else {
                // WASM
                if (cv instanceof Promise) {
                    cv = await cv;
                    console.log(cv.getBuildInformation());
                    onloadCallback();
                } else {
                    cv['onRuntimeInitialized'] = () => {
                        console.log(cv.getBuildInformation());
                        onloadCallback();
                    }
                }
            }
        });
        script.addEventListener('error', () => {
            self.printError('Failed to  load ' + OPENCV_URL);
        });
        script.src = OPENCV_URL;
        let node = document.getElementsByTagName('script')[0];
        node.parentNode.insertBefore(script, node);
    };

    this.createFileFromUrl = (path, url, callback) => {
        let request = new XMLHttpRequest();
        request.open('GET', url, true);
        request.responseType = 'arraybuffer';
        request.onload = ev => {
            if (request.readyState === 4) {
                if (request.status === 200) {
                    let data = new Uint8Array(request.response);
                    cv.FS_createDataFile('/', path, data, true, false, false);
                    callback();
                } else {
                    self.printError('Failed to load ' + url + ' status : ' + request.status);
                }
            }
        };
        request.send();
    };


    this.processVideo = (input, output, classifierModel, emotionResultId, chart) => {
        try {
            this.clearError();
            this.classifyEmotion(input, output, classifierModel, emotionResultId, chart);
        } catch (err) {
            this.printError(err);
        }
    };

    this.clearError = () => {
        this.errorOutput.innerHTML = '';
    };

    this.printError = (err) => {
        if (typeof err === 'undefined') {
            err = '';
        } else if (typeof err === 'number') {
            if (!isNaN(err)) {
                if (typeof cv !== 'undefined') {
                    err = 'Exception: ' + cv.exceptionFromPtr(err).msg;
                }
            }
        } else if (typeof err === 'string') {
            let ptr = Number(err.split(' ')[0]);
            if (!isNaN(ptr)) {
                if (typeof cv !== 'undefined') {
                    err = 'Exception: ' + cv.exceptionFromPtr(ptr).msg;
                }
            }
        } else if (err instanceof Error) {
            err = err.stack.replace(/\n/g, '<br>');
        }
        this.errorOutput.innerHTML = err;
    };

    const onVideoCanPlay = () => {
        if (self.onCameraStartedCallback) {
            self.onCameraStartedCallback(self.stream, self.video);
        }
    };

    this.startCamera = (resolution, callback, videoId) => {
        const constraints = {
            'qvga': {width: {exact: 320}, height: {exact: 240}},
            'vga': {width: {exact: 640}, height: {exact: 480}}
        };
        let video = document.getElementById(videoId);
        if (!video) {
            video = document.createElement('video');
        }

        let videoConstraint = constraints[resolution];
        if (!videoConstraint) {
            videoConstraint = true;
        }

        navigator.mediaDevices.getUserMedia({video: videoConstraint, audio: false})
            .then(stream => {
                video.srcObject = stream;
                video.play();
                self.video = video;
                self.stream = stream;
                self.onCameraStartedCallback = callback;
                video.addEventListener('canplay', onVideoCanPlay, false);
            })
            .catch(err => {
                self.printError('Camera Error: ' + err.name + ' ' + err.message);
            });
    };

    this.stopCamera = () => {
        if (this.video) {
            this.video.pause();
            this.video.srcObject = null;
            this.video.removeEventListener('canplay', onVideoCanPlay);
        }
        if (this.stream) {
            this.stream.getVideoTracks()[0].stop();
        }
    };

    this.classifyEmotion = (inputId, outputId, classifierModel, emotionResultId, chart) => {
        let video = document.getElementById(inputId);
        let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
        let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
        let gray = new cv.Mat();
        let cap = new cv.VideoCapture(video);
        let faces = new cv.RectVector();
        let classifier = new cv.CascadeClassifier();
        let frameQueue = [];

        // load pre-trained classifiers
        classifier.load(classifierModel);

        const FPS = 3;

        const processVideo = () => {
            try {
                if (!streaming) {
                    // clean and stop.
                    src.delete();
                    dst.delete();
                    gray.delete();
                    faces.delete();
                    classifier.delete();
                    document.getElementById(emotionResultId).innerHTML = ''
                    frameQueue = [];
                    return;
                }
                let begin = Date.now();
                // start processing.
                cap.read(src);
                src.copyTo(dst);
                cv.cvtColor(dst, gray, cv.COLOR_RGBA2GRAY, 0);
                // detect faces.
                classifier.detectMultiScale(gray, faces, 1.1, 3, 0);
                // draw faces.
                for (let i = 0; i < faces.size(); ++i) {
                    let face = faces.get(i);
                    let cropped;
                    let padding = 10;
                    if (face.x - padding < 0 || face.y - padding < 0 || face.width + padding > gray.width || face.height + padding > gray.height) {
                        cropped = gray.roi(new cv.Rect(face.x, face.y, face.width, face.height));
                    } else {
                        cropped = gray.roi(new cv.Rect(face.x - padding, face.y - padding, face.width + padding, face.height + padding));
                    }
                    cv.resize(cropped, cropped, (new cv.Size(48, 48)), 0, 0, cv.INTER_AREA);
                    // cv.normalize(cropped, cropped, 50, 350, cv.NORM_MINMAX)
                    frameQueue.push(cropped)
                    if (frameQueue.length > 3) {
                        frameQueue.shift();
                        let classifyAndShow = async (frameQueue) => {
                            //output frames
                            let mergedArray;
                            mergedArray = new Uint8Array(48 * 48 * 3);
                            let offset = 0
                            for (let i = 0; i < frameQueue.length; i++) {
                                mergedArray.set(frameQueue[i].data, offset);
                                offset += frameQueue[i].data.length;
                            }
                            let imagesMatrix = cv.matFromArray(48 * 3, 48, cv.CV_8UC1, mergedArray);

                            // classify emotion
                            let emotionsResponse = await this.requestEmotion(frameQueue.map(x => Array.from(x.data)));
                            return {matrix: imagesMatrix, emotions: emotionsResponse}
                        }

                        classifyAndShow(frameQueue).then(result => {
                                cv.imshow(outputId, result.matrix);
                                const container = document.getElementById(emotionResultId);
                                container.innerHTML = '';

                                let chartData = chart.data;
                                chartData.labels.push('.');
                                result.emotions.result.forEach(item => {
                                    let paragraph = document.createElement("p");
                                    paragraph.innerHTML = item.emotion + " : " + item.value.toLocaleString('en-US', {
                                        maximumFractionDigits: 6,
                                        useGrouping: false
                                    });
                                    container.appendChild(paragraph);

                                    chartData.datasets
                                        .find(dataset => item.emotion.toLowerCase().includes(dataset.label.toLowerCase()))
                                        .data.push(item.value)
                                })
                                chart.update();
                            }
                        )
                    }
                }
                // schedule the next one.
                let delay = 1000 / FPS - (Date.now() - begin);
                setTimeout(processVideo, delay);
            } catch (err) {
                utils.printError(err);
            }
        };

        // schedule the first one.
        setTimeout(processVideo, 0);
    };

    this.requestEmotion = async (arrays) => {
        try {
            const response = await fetch('/', {
                method: 'POST',
                port: 3000,
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    images: arrays
                })
            });
            return await response.json();
        } catch (error) {
            console.error('Error:', error);
            return {}
        }
    }
}