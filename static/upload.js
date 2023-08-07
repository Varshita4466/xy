function uploadImage() {
  const fileInput = document.getElementById("image-input");
  const file = fileInput.files[0];

  if (!file) {
      alert("Please select an image.");
      return;
  }

  const formData = new FormData();
  formData.append("image", file);

  fetch("/predict", {
      method: "POST",
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      showPredictionResult(data.result, data.probability);
  })
  .catch(error => console.error("Error:", error));
}

function showPredictionResult(predictedDisease, probability) {
  const predictionSection = document.getElementById("prediction-result");
  const predictedDiseaseElement = document.getElementById("predicted-disease");
  const probabilityElement = document.getElementById("probability");

  predictedDiseaseElement.textContent = "Predicted Disease: " + predictedDisease;
  probabilityElement.textContent = "Probability: " + (probability * 100).toFixed(2) + "%";

  predictionSection.style.display = "block";
}
