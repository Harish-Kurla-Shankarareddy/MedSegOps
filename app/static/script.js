const form =
    document.getElementById("upload-form");

const fileInput =
    document.getElementById("file-input");

const selectedFile =
    document.getElementById("selected-file");

const segmentButton =
    document.getElementById("segment-button");

const loading =
    document.getElementById("loading");

const results =
    document.getElementById("results");

const errorBox =
    document.getElementById("error");

const errorMessage =
    document.getElementById("error-message");

const visualization =
    document.getElementById("visualization");

const downloadLink =
    document.getElementById("download-link");

const resultMessage =
    document.getElementById("result-message");


/* ----------------------------------------
   Display selected file name
----------------------------------------- */

fileInput.addEventListener(
    "change",
    () => {

        if (
            fileInput.files.length > 0
        ) {

            selectedFile.textContent =
                fileInput.files[0].name;

        } else {

            selectedFile.textContent =
                "No file selected";

        }

    }
);


/* ----------------------------------------
   Submit segmentation
----------------------------------------- */

form.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();


        // Reset UI

        results.classList.add("hidden");

        errorBox.classList.add("hidden");

        loading.classList.remove("hidden");

        segmentButton.disabled = true;

        segmentButton.textContent =
            "Segmenting...";


        try {

            const file =
                fileInput.files[0];


            const formData =
                new FormData();

            formData.append(
                "file",
                file
            );


            const response =
                await fetch(
                    "/segment",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Segmentation failed"
                );

            }


            // Add timestamp to avoid
            // browser image caching

            const timestamp =
                new Date().getTime();


            visualization.src =
                data.visualization_url +
                "?t=" +
                timestamp;


            downloadLink.href =
                data.segmentation_url;


            resultMessage.textContent =
                data.message;


            results.classList.remove(
                "hidden"
            );


        } catch (error) {

            errorMessage.textContent =
                error.message;

            errorBox.classList.remove(
                "hidden"
            );

        } finally {

            loading.classList.add(
                "hidden"
            );

            segmentButton.disabled =
                false;

            segmentButton.textContent =
                "Segment CT";

        }

    }
);