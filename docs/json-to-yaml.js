const fs = require('fs');
const path = require('path');
const yaml = require('yaml');

const json = JSON.parse(fs.readFileSync(path.join(__dirname, 'elements.json')).toString());

fs.writeFileSync(path.join(__dirname, 'elements.yaml'), yaml.stringify(json).replace(/- /g, "\n- ").trim());