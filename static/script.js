document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("newsForm");
  const textArea = document.getElementById("newsText");
  const fileInput = document.getElementById("newsFile");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const btnText = document.getElementById("btnText");
  const charCount = document.getElementById("charCount");
  const results = document.getElementById("results");
  const sampleBtn = document.getElementById("sampleBtn");

  // Character counter
  textArea.addEventListener("input", () => {
    const count = textArea.value.length;
    charCount.textContent = `${count} / 45,000`;
    charCount.style.color = count > 45000 ? "#dc3545" : "#6c757d";
  });

  // Sample article button
  sampleBtn.addEventListener("click", () => {
    const sample = `Breaking News: Local Scientists Claim to Have Discovered Method to Turn Water into Gold

In a groundbreaking announcement that has sent shockwaves through the scientific community, researchers at the fictitious Institute of Alchemy have claimed to have successfully developed a revolutionary process that can transform ordinary water into pure gold. Dr. Sarah Mitchell, the lead researcher, stated during a press conference yesterday that their team has been working on this project for over two years in complete secrecy.

"We have finally cracked the code that alchemists have been trying to solve for centuries," Dr. Mitchell announced to a room full of skeptical journalists. "Our proprietary process, which we're calling 'Hydro-Aurum Transformation,' can convert H2O molecules into gold atoms through a series of quantum manipulations that we cannot fully disclose due to patent pending status."

This story will be updated as more information becomes available.`;

    textArea.value = sample;
    textArea.dispatchEvent(new Event("input")); // Trigger input event to update char count
  });

  // Clear file when text is entered
  textArea.addEventListener("input", () => {
    if (textArea.value.trim()) {
      fileInput.value = "";
    }
  });

  // Clear text when file is selected
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) {
      textArea.value = "";
      charCount.textContent = "0 / 45,000";
    }
  });

  // Form submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const text = textArea.value.trim();
    const file = fileInput.files[0];

    if (!text && !file) {
      showError("Please enter text or select a file to analyze.");
      return;
    }

    setLoading(true);
    results.innerHTML = "";

    try {
      const formData = new FormData();
      if (file) {
        formData.append("file", file);
      } else {
        formData.append("text", text);
      }

      // Note: The endpoint is now '/analyze'
      const response = await fetch("/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Analysis failed");
      }

      const data = await response.json();
      showResults(data);
    } catch (error) {
      showError(error.message);
    } finally {
      setLoading(false);
    }
  });

  function setLoading(loading) {
    analyzeBtn.disabled = loading;
    if (loading) {
      btnText.innerHTML = '<span class="loading-spinner"></span> Analyzing...';
    } else {
      btnText.textContent = "Analyze News";
    }
  }

  function showError(message) {
    results.innerHTML = `
            <div class="result-container">
                <div class="alert alert-danger" role="alert">
                    <h4 class="alert-heading">Error</h4>
                    <p class="mb-0">${message}</p>
                </div>
            </div>`;
  }

  function showResults(data) {
    // Your model returns 1 for REAL and 0 for FAKE.
    const isFake = data.prediction === "Looking FAKE ⚠ News📰";
    const confidence = (data.confidence * 100).toFixed(1);

    results.innerHTML = `
            <div class="result-container ${
              isFake ? "fake-result" : "real-result"
            }">
                <div class="result-header">
                    <h3>${isFake ? "FAKE NEWS" : "LIKELY REAL"}</h3>
                    <p class="mb-0">Confidence: ${confidence}%</p>
                </div>
                <div class="result-body">
                    <h5>Confidence Score</h5>
                    <div class="confidence-bar">
                        <div class="progress-bar ${
                          isFake ? "bg-danger" : "bg-success"
                        }" 
                             style="width: ${confidence}%; height: 100%"></div>
                    </div>
                    <div class="text-center mt-2">
                        <strong>${confidence}%</strong>
                    </div>
                    <div class="alert alert-info mt-3" role="alert">
                        <small><strong>Note:</strong> This analysis is based on machine learning patterns and should be used as a reference only.</small>
                    </div>
                </div>
            </div>`;
  }
});
