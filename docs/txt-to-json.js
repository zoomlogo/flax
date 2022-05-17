const fs = require('fs');
const path = require('path');

const txt = fs.readFileSync(path.join(__dirname, 'elements.txt')).toString();

const lines = txt.split(/\r?\n/);
const split = lines.map(x => x.split(/ +/)).map(x => [x[0], x[1], x.slice(2).join(' ')]);

const json = JSON.stringify(split.map(x => ({element: x[0], arity: x[1], description: x[2]})));

fs.writeFileSync(path.join(__dirname, 'elements.json'), json);