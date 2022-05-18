const fs = require('fs');
const path = require('path');
const yaml = require('yaml');

const y = yaml.parse(fs.readFileSync(path.join(__dirname, 'elements.yaml')).toString());

fs.writeFileSync(path.join(__dirname, 'elements.json'), JSON.stringify(y, null, "  "));