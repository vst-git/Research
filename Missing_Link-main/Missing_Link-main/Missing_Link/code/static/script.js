let selectedFile;

document
  .getElementById("file-input")
  .addEventListener("change", function (event) {
    selectedFile = event.target.files[0];
    const videoPreview = document.getElementById("preview");
    if (selectedFile) {
      videoPreview.src = URL.createObjectURL(selectedFile);
      videoPreview.style.display = "block";
      document.getElementById("upload-btn").disabled = false;
    } else {
      videoPreview.style.display = "none";
      document.getElementById("upload-btn").disabled = true;
    }
  });

document.getElementById("select-btn").addEventListener("click", function () {
  document.getElementById("file-input").click();
});

document.getElementById("upload-btn").addEventListener("click", function () {
  if (selectedFile) {
    document.getElementsByClassName(
      "indeterminate-progress-bar"
    )[0].style.visibility = "visible";
    const uploadAddress = "/upload";
    const formData = new FormData();
    formData.append("video", selectedFile);

    fetch(uploadAddress, {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (response.redirected) {
          document.getElementsByClassName(
            "indeterminate-progress-bar"
          )[0].style.visibility = "hidden";
          window.location.href = response.url;
        } else {
          console.error("Redirection failed.");
        }
      })
      .catch((error) => {
        console.error("Upload failed:", error);
      });
  }
});
