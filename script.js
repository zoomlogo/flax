console.log("Loaded!");

const getJSON = async url => {
  const response = await fetch(url);
  return response.json();
}

/* load json */
console.log("Fetching Data...");
getJSON("https://raw.githubusercontent.com/PyGamer0/flax/main/docs/elements.json").then(elements => {
  let el = document.getElementById("el");
  elements.forEach(element => {
    let tr = document.createElement('tr');
    for (let key of ['element', 'description', 'arity-type']) {
      let td = document.createElement('td')
      td.innerText = element[key];
      tr.appendChild(td);
    }
    tr.addEventListener("click", () => {
      console.log("Copying...");
      navigator.clipboard.writeText(element["element"]).then(() => { console.log("Copied!"); });
    });
    el.appendChild(tr);
  });
});
console.log("Fetched!");

/* search function */
const filter_search = () => {
  let bar = document.getElementById("search-bar");
  let filter = bar.value.toLowerCase();
  let table = document.getElementById("el");
  let rows = table.getElementsByTagName("tr");

  for (let i = 0; i < rows.length; i++) {
    let row = rows[i];
    let d = row.getElementsByTagName("td")[1];
    if (d) {
      let value = (d.textContent || d.innerText).toLowerCase();
      if (value.includes(filter)) {
        row.hidden = false;
      } else {
        row.hidden = true;
      }
    }
  }
}
