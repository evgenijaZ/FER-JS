const express = require('express');
const routes = require('./routes/index');
const bodyParser = require('body-parser');

const app = express();
app.use('/', routes);
app.use(express.static("public"));
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
module.exports = app;