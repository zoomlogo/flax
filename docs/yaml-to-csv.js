const csv = require('json2csv');
const yaml = require('yaml');
const fs = require('fs');
const path = require('path');

const json = yaml.parse(fs.readFileSync(path.join(__dirname, "elements.yaml")).toString());

fs.writeFileSync(path.join(__dirname, "elements.csv"), csv.parse(json));