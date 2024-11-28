const express = require('express');
const app = express();
const path = require('path');


app.set('view engine', 'ejs');
app.set('views', path.join(__dirname, 'views'));

// Serve static files like CSS and JS
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.static('public'));
// Home Route


// Function to handle AI Bot chat interaction
function startAiBotInteraction() {
    const userMessage = prompt("Ask the AI Bot anything:");

    if (userMessage) {
        fetch("http://localhost:5000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message: userMessage }),
        })
        .then((response) => response.json())
        .then((data) => {
            // Display AI bot's response on the page or as an alert
            alert(`AI Bot: ${data.response}`);
            const aiResponseElement = document.getElementById("ai-bot-response");
            if (aiResponseElement) {
                aiResponseElement.innerHTML = `<strong>AI Bot:</strong> ${data.response}`;
            }
        })
        .catch((error) => {
            console.error("Error:", error);
            alert("Failed to communicate with the AI Bot.");
        });
    }
}

// Function to start voice interaction
function startVoiceInteraction() {
    const responseSection = document.getElementById('ai-bot-response');
    responseSection.innerHTML = `<p>Listening...</p>`;

    // Fetch backend response
    fetch("http://localhost:5000/start-voice-interaction", {
        method: "GET",
    })
    .then((response) => response.json())
    .then((data) => {
        responseSection.innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
    })
    .catch((error) => {
        console.error("Error:", error);
        responseSection.innerHTML += `<p>Sorry, there was an error processing your request.</p>`;
    });
}


app.get('/', (req, res) => {
    res.render('home');
});

app.get('/about', (req, res) => {
    res.render('about');
});

app.get('/start-voice-interaction', (req, res) => {
    res.json({ response: "Hello! How can I help you?" });
});

app.get('/services', (req, res) => {
    res.render('services');
});

app.get('/contact', (req, res) => {
    res.render('contact');
});

app.post("/chat", (req, res) => {
  const { message } = req.body;

  if (!message) {
    return res.status(400).json({ response: "Please provide a valid message." });
  }

  const response = "AI Bot says: " + message;
  res.json({ response });
});

app.post("/process-voice-input", (req, res) => {
    const userInput = req.body.input;

    // Process the input (you can integrate AI or predefined responses here)
    let botResponse = "I heard: " + userInput;

    // Send the response back to the frontend
    res.json({ response: botResponse });
});


const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
