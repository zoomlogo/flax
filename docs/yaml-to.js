const csv = require("json2csv");
const yaml = require('yaml');
const fs = require('fs');
const path = require('path');
const columnify = require('columnify');

const json = yaml.parse(fs.readFileSync(path.join(__dirname, "elements.yaml")).toString());

fs.writeFileSync(path.join(__dirname, "elements.csv"), csv.parse(json));  // csv
fs.writeFileSync(path.join(__dirname, 'elements.json'), JSON.stringify(y, null, "  "));  // json
fs.writeFileSync(path.join(__dirname, "elements.txt"), columnify(json).split('\n').map(x => x.trimEnd()).join('\n'));  // txt
