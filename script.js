console.log("Loaded!");

const getJSON = async url => {
  const response = await fetch(url);
  return response.json();
}

/* load json */
console.log("Fetching Data...");
getJSON("https://raw.githubusercontent.com/PyGamer0/flax/main/docs/elements.json").then(elements => {
  var el = document.getElementById("el");
  elements.forEach(element => {
    el.insertAdjacentHTML('beforeend', "<tr><td>" + element["element"] + "</td><td>" + element["description"] + "</td></tr>");
  });
});
