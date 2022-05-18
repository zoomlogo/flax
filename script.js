console.log("Loaded!");

const getJSON = async url => {
    const response = await fetch(url);
    return response.json();
}

/* load json */
console.log("Fetching Data...");
getJSON("https://raw.githubusercontent.com/PyGamer0/flax/main/docs/elements.json").then(
    data => { console.log(data); }
)
