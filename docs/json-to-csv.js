const csv = require('json2csv');
const fs = require('fs');
const path = require('path');

const json = JSON.parse(fs.readFileSync(path.join(__dirname, "elements.json")).toString());

fs.writeFileSync(path.join(__dirname, "elements.csv"), csv.parse(json));