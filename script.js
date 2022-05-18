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
    el.insertAdjacentHTML('beforeend', "<tr><td>" + element["element"] + "</td><td>" + element["description"] + "</td><td>" + element["arity-type"] + "</td></tr>");
  });
});

/* search function */
const filter_search = () => {
  let bar = document.getElementById("search-bar");
  let filter = bar.value.toLowerCase();
  let table = document.getElementById("el");
  let rows = table.getElementsByTagName("tr");

  rows.forEach(row => {
    let d = row.getElementsByTagName("td")[1];
    if (d) {
      let value = (d.textContent || d.innerText).toLowerCase();
      if (value.indexOf(filter) > -1)
        row.style.display = "";
      else
        row.style.display = "none";
    }
  })
}
