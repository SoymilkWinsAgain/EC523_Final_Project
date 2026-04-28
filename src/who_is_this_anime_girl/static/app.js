const statusText = document.querySelector("#statusText");
const refreshStatus = document.querySelector("#refreshStatus");
const enrollForm = document.querySelector("#enrollForm");
const queryForm = document.querySelector("#queryForm");
const textQueryForm = document.querySelector("#textQueryForm");
const enrollResult = document.querySelector("#enrollResult");
const queryResult = document.querySelector("#queryResult");
const textQueryResult = document.querySelector("#textQueryResult");
const rebuildIndex = document.querySelector("#rebuildIndex");
const identityList = document.querySelector("#identityList");
const matches = document.querySelector("#matches");

function setBusy(element, busy) {
  element.disabled = busy;
}

function showMessage(element, text, isError = false) {
  element.textContent = text;
  element.classList.toggle("error", isError);
}

function renderMatches(data, targetResult) {
  if (data.matches.length === 0) {
    showMessage(targetResult, "No matches found.");
    return;
  }
  const mode = data.mode ? ` (${data.mode})` : "";
  showMessage(targetResult, `Found ${data.matches.length} match(es)${mode}.`);
  for (const match of data.matches) {
    const card = document.createElement("article");
    card.className = "match-card";
    const image = document.createElement("img");
    image.alt = match.identity;
    image.src = match.gallery_url || "";
    const detail = document.createElement("div");
    const title = document.createElement("strong");
    title.textContent = match.identity;
    const score = document.createElement("span");
    score.textContent = `Similarity: ${match.score.toFixed(4)}`;
    const path = document.createElement("span");
    path.textContent = match.path;
    detail.append(title, score, path);
    card.append(image, detail);
    matches.appendChild(card);
  }
}

async function readJson(response) {
  const data = await response.json();
  if (!response.ok || data.ok === false) {
    throw new Error(data.error || `Request failed with status ${response.status}`);
  }
  return data;
}

async function loadStatus() {
  const data = await fetch("/api/status").then(readJson);
  const model = data.checkpoint_exists ? "model ready" : "model missing";
  const index = data.index_exists ? "index ready" : "index missing";
  statusText.textContent = `${model}; ${index}`;
  identityList.innerHTML = "";
  const entries = Object.entries(data.identities || {});
  if (entries.length === 0) {
    identityList.textContent = "No gallery identities enrolled yet.";
    return;
  }
  for (const [identity, count] of entries) {
    const pill = document.createElement("div");
    pill.className = "identity-pill";
    pill.textContent = `${identity}: ${count}`;
    identityList.appendChild(pill);
  }
}

refreshStatus.addEventListener("click", () => {
  loadStatus().catch((error) => {
    statusText.textContent = error.message;
  });
});

enrollForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(enrollForm.querySelector("button[type='submit']"), true);
  showMessage(enrollResult, "Uploading images...");
  try {
    const formData = new FormData(enrollForm);
    const data = await fetch("/api/enroll", { method: "POST", body: formData }).then(readJson);
    showMessage(enrollResult, `Saved ${data.count} image(s) for ${data.identity}.`);
    enrollForm.reset();
    await loadStatus();
  } catch (error) {
    showMessage(enrollResult, error.message, true);
  } finally {
    setBusy(enrollForm.querySelector("button[type='submit']"), false);
  }
});

rebuildIndex.addEventListener("click", async () => {
  setBusy(rebuildIndex, true);
  showMessage(enrollResult, "Rebuilding gallery index...");
  try {
    const data = await fetch("/api/rebuild", { method: "POST" }).then(readJson);
    showMessage(enrollResult, `Indexed ${data.indexed_images} gallery image(s).`);
    await loadStatus();
  } catch (error) {
    showMessage(enrollResult, error.message, true);
  } finally {
    setBusy(rebuildIndex, false);
  }
});

queryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(queryForm.querySelector("button[type='submit']"), true);
  showMessage(queryResult, "Searching...");
  matches.innerHTML = "";
  try {
    const formData = new FormData(queryForm);
    const data = await fetch("/api/query", { method: "POST", body: formData }).then(readJson);
    renderMatches(data, queryResult);
  } catch (error) {
    showMessage(queryResult, error.message, true);
  } finally {
    setBusy(queryForm.querySelector("button[type='submit']"), false);
  }
});

textQueryForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  setBusy(textQueryForm.querySelector("button[type='submit']"), true);
  showMessage(textQueryResult, "Searching...");
  matches.innerHTML = "";
  try {
    const formData = new FormData(textQueryForm);
    const data = await fetch("/api/query_text", { method: "POST", body: formData }).then(readJson);
    renderMatches(data, textQueryResult);
  } catch (error) {
    showMessage(textQueryResult, error.message, true);
  } finally {
    setBusy(textQueryForm.querySelector("button[type='submit']"), false);
  }
});

loadStatus().catch((error) => {
  statusText.textContent = error.message;
});
