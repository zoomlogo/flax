const yaml = require('yaml');
const fs = require('fs');
const path = require('path');
const columnify = require('columnify');

const json = yaml.parse(fs.readFileSync(path.join(__dirname, "elements.yaml")).toString());

fs.writeFileSync(path.join(__dirname, "elements.txt"), columnify(json).split('\n').map(x => x.trimEnd()).join('\n'));