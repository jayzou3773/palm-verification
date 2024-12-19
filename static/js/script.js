document.addEventListener("DOMContentLoaded", function () {

    const uploadForm = document.getElementById("upload-form");
    const checkStatusBtn = document.getElementById("check-status-btn");
    const compareForm = document.getElementById("compare-form");

    // Handle image upload
    uploadForm.addEventListener("submit", function (event) {
        event.preventDefault();

        const formData = new FormData(uploadForm);
        const imageId = document.getElementById("image_id").value;

        fetch(`/upload/${imageId}`, {
            method: "POST",
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                document.getElementById("upload-status").innerHTML = data.message;
            })
            .catch(error => {
                document.getElementById("upload-status").innerHTML = 'Error uploading images.';
                console.error("Error uploading images:", error);
            });
    });

    // Check process status
    checkStatusBtn.addEventListener("click", function () {
        fetch('/check', {
            method: "GET",
        })
            .then(response => response.json())
            .then(data => {
                const statusMessage = data.status === 'completed' ? "All images have been processed." :
                    data.status === 'processing' ? "Images are being processed." :
                        "Image processing failed.";
                document.getElementById("process-status").innerHTML = statusMessage;
            })
            .catch(error => {
                document.getElementById("process-status").innerHTML = 'Error checking status.';
                console.error("Error checking status:", error);
            });
    });

    // Handle image comparison
    compareForm.addEventListener("submit", function (event) {
        event.preventDefault();

        const formData = new FormData(compareForm);

        fetch('/compare', {
            method: "POST",
            body: formData,
        })
            .then(response => response.json())
            .then(data => {
                document.getElementById("comparison-result").innerHTML = data.message;
            })
            .catch(error => {
                document.getElementById("comparison-result").innerHTML = 'Error comparing images.';
                console.error("Error comparing images:", error);
            });
    });

});
