async function sendMessage() {
  const input = document.getElementById("userInput");
  const text = input.value.trim();
  if (!text) return;

  appendMessage("user", text);
  input.value = "";

  const chatBox = document.getElementById("chat-box");
  const history = Array.from(chatBox.children).map(div => ({
    role: div.classList.contains("user") ? "user" : "assistant",
    content: div.textContent
  }));

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages: history })
  });

  const data = await res.json();
  appendMessage("bot", data.reply);
}

function appendMessage(role, text) {
  const chatBox = document.getElementById("chat-box");
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}
