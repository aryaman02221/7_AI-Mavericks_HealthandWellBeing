const images = document.querySelectorAll('.lazy-load');
const observer = new IntersectionObserver((entries, observer) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.src = entry.target.getAttribute('data-src');
      observer.unobserve(entry.target);
    }
  });
});
images.forEach(image => {
  observer.observe(image);
});


document.addEventListener("DOMContentLoaded", () => {
  const images = document.querySelectorAll(".image-container");

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.animationDelay = `${Math.random() * 0.5}s`; // Random delay for staggered effect
          entry.target.classList.add("visible");
        }
      });
    },
    { threshold: 0.5 }
  );

  images.forEach((image) => observer.observe(image));
});


window.onload = function() {
  const services = document.querySelectorAll('.service');
  let index = 0;

  function showService() {
    if (index < services.length) {
      services[index].style.opacity = 1;
      services[index].style.transform = 'translateY(0)';
      index++;
    }
  }

  setInterval(showService, 3000);

  const serviceElements = document.querySelectorAll('.service');

  serviceElements.forEach(service => {
    service.addEventListener('mouseover', function() {
      serviceElements.forEach(s => {
        if (s !== service) {
          s.classList.add('service-blurred');
        }
      });
      service.querySelector('p').style.display = 'block';
    });

    service.addEventListener('mouseleave', function() {
      serviceElements.forEach(s => {
        s.classList.remove('service-blurred');
      });
      service.querySelector('p').style.display = 'none';
    });
  });
};


document.addEventListener("DOMContentLoaded", () => {
    const services = document.querySelectorAll(".service");
    const descriptionSection = document.querySelector(".service-description p");

    services.forEach((service) => {
        service.addEventListener("mouseenter", () => {
            const description = service.getAttribute("data-description");
            descriptionSection.textContent = description;
        });

        service.addEventListener("mouseleave", () => {
            descriptionSection.textContent = "Hover over a service to see the description.";
        });
    });
});


function startAiBotInteraction() {
  const userMessage = prompt("Ask the AI Bot anything:");

  if (userMessage) {
    // Send the user's message to the backend using fetch
    fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: userMessage }),
    })
      .then((response) => response.json())
      .then((data) => {
        // Display AI bot's response in an alert or on the page
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



// document.getElementById("sendButton").addEventListener("click", function () {
//     const userInput = document.getElementById("userInput").value;
//
//     fetch("/chat", {
//         method: "POST",
//         headers: {
//             "Content-Type": "application/json",
//         },
//         body: JSON.stringify({ message: userInput }),
//     })
//         .then((response) => response.json())
//         .then((data) => {
//             const chatOutput = document.getElementById("chatOutput");
//             chatOutput.innerHTML += `<p>User: ${userInput}</p>`;
//             chatOutput.innerHTML += `<p>Bot: ${data.response}</p>`;
//         })
//         .catch((error) => console.error("Error:", error));
// });
// Text Interaction
document.getElementById("shareButton").addEventListener("click", async () => {
    const userMessage = document.getElementById("userInput").value;

    if (userMessage.trim()) {
        const response = await fetch("http://localhost:5000/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: userMessage })
        });

        const result = await response.json();
        document.getElementById("responseOutput").innerText = result.response || "No response received.";
    } else {
        alert("Please enter a message.");
    }
});





function startVoiceInteraction() {
    // Inform the user about starting the voice interaction
    const responseSection = document.getElementById('ai-bot-response');
    responseSection.innerHTML = `<p>Listening...</p>`;

    // Start the Speech Recognition API
    const recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "en-US";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.start();

    recognition.onresult = function(event) {
        const transcript = event.results[0][0].transcript;
        responseSection.innerHTML += `<p><strong>You:</strong> ${transcript}</p>`;

        // Send the transcript to the backend for processing
        fetch("/process-voice-input", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ input: transcript })
        })
        .then((response) => response.json())
        .then((data) => {
            responseSection.innerHTML += `<p><strong>Bot:</strong> ${data.response}</p>`;
        })
        .catch((error) => {
            console.error("Error:", error);
            responseSection.innerHTML += `<p>Sorry, there was an error processing your request.</p>`;
        });
    };

    recognition.onerror = function(event) {
        console.error("Speech recognition error", event.error);
        responseSection.innerHTML += `<p>Sorry, I couldn't understand your speech.</p>`;
    };

    recognition.onend = function() {
        // This is triggered when the recognition ends, regardless of success
        console.log("Speech recognition has ended.");
    };
}
