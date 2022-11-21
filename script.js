console.log("Loaded!");

const getJSON = async (url) => {
  const response = await fetch(url);
  return response.json();
};

/* load json */
console.log("Fetching Data...");
getJSON(
  "https://raw.githubusercontent.com/PyGamer0/flax/main/docs/elements.json"
).then((elements) => {
  let el = document.getElementById("el");
  elements.forEach((element) => {
    let tr = document.createElement("tr");

    let tooltip = document.createElement("span");
    tooltip.innerHTML = "Copied!";
    tooltip.classList.add("tooltip"); //idk why but this works
    tr.appendChild(tooltip);

    for (let key of ["element", "description", "type"]) {
      let td = document.createElement("td");
      td.innerText = element[key];
      tr.appendChild(td);
    }
    tr.addEventListener("click", () => {
      console.log("Copying...");
      tooltip.style.visibility = "visible";
      setTimeout(() => {
        fade(tooltip);
      }, 300);

      navigator.clipboard.writeText(element["element"]).then(() => {
        console.log("Copied!");
      });
    });
    el.appendChild(tr);
  });
});
console.log("Fetched!");

/* fadeout function for tooltip */
function fade(elem) {
  var fadeEffect = setInterval(function () {
    if (!elem.style.opacity) {
      elem.style.opacity = 1;
    }
    if (elem.style.opacity > 0.1) {
      elem.style.opacity -= 0.1;
    } else {
      clearInterval(fadeEffect);
      elem.style.opacity = "";
      elem.style.visibility = "";
    }
  }, 50);
}

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
};
