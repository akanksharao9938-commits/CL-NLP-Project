// ============================================================
// EXAM PATTERN RAG - FRONTEND JAVASCRIPT
// ============================================================

const API_URL = "http://127.0.0.1:8000";


// ============================================================
// DOM ELEMENTS
// ============================================================

const queryInput = document.getElementById("queryInput");
const analyzeBtn = document.getElementById("analyzeBtn");

const loading = document.getElementById("loading");
const errorMessage = document.getElementById("errorMessage");

const resultSection = document.getElementById("resultSection");

const questionCount = document.getElementById("questionCount");
const yearsFound = document.getElementById("yearsFound");
const mainTopic = document.getElementById("mainTopic");

const ragAnswer = document.getElementById("ragAnswer");
const questionsContainer =
    document.getElementById("questionsContainer");

const forecastBtn =
    document.getElementById("forecastBtn");

const forecastLoading =
    document.getElementById("forecastLoading");

const forecastContainer =
    document.getElementById("forecastContainer");


// ============================================================
// PAGE LOAD
// ============================================================

document.addEventListener("DOMContentLoaded", () => {

    console.log("ExamPatternRAG frontend loaded.");

    console.log("API URL:", API_URL);

    console.log(
        "Analyze button:",
        analyzeBtn
    );

    console.log(
        "Query input:",
        queryInput
    );

    checkBackend();

});


// ============================================================
// CHECK BACKEND
// ============================================================

async function checkBackend() {

    try {

        console.log(
            "Checking backend..."
        );

        const response =
            await fetch(API_URL);

        if (!response.ok) {

            throw new Error(
                `Backend returned ${response.status}`
            );

        }

        const data =
            await response.json();

        console.log(
            "Backend connected successfully:",
            data
        );

    } catch (error) {

        console.error(
            "Backend connection failed:",
            error
        );

        showError(
            "Could not connect to backend. Make sure FastAPI is running on http://127.0.0.1:8000"
        );

    }

}


// ============================================================
// ANALYZE BUTTON
// ============================================================

if (analyzeBtn) {

    analyzeBtn.addEventListener(
        "click",
        handleQuery
    );

} else {

    console.error(
        "ERROR: Analyze button not found."
    );

}


// ============================================================
// ENTER KEY
// ============================================================

if (queryInput) {

    queryInput.addEventListener(
        "keydown",
        function(event) {

            if (event.key === "Enter") {

                event.preventDefault();

                handleQuery();

            }

        }
    );

}


// ============================================================
// HANDLE QUERY
// ============================================================

async function handleQuery() {

    console.log(
        "Analyze button clicked."
    );


    if (!queryInput) {

        console.error(
            "Query input not found."
        );

        return;

    }


    const query =
        queryInput.value.trim();


    console.log(
        "User query:",
        query
    );


    // --------------------------------------------------------
    // Empty query
    // --------------------------------------------------------

    if (!query) {

        showError(
            "Please enter a topic or question."
        );

        return;

    }


    // --------------------------------------------------------
    // Start loading
    // --------------------------------------------------------

    setLoading(true);

    clearError();


    if (resultSection) {

        resultSection.classList.add(
            "hidden"
        );

    }


    try {

        console.log(
            "Sending POST request to:",
            `${API_URL}/query`
        );


        // ====================================================
        // SEND QUERY TO FASTAPI
        // ====================================================

        const response =
            await fetch(
                `${API_URL}/query`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        query: query,
                        top_k: 5
                    })
                }
            );


        console.log(
            "Backend HTTP status:",
            response.status
        );


        // ====================================================
        // READ RESPONSE
        // ====================================================

        const data =
            await response.json();


        console.log(
            "Backend response:",
            data
        );


        // ====================================================
        // HANDLE ERROR
        // ====================================================

        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Backend returned an error."
            );

        }


        // ====================================================
        // DISPLAY RESULTS
        // ====================================================

        displayResults(data);


    } catch (error) {

        console.error(
            "Analyze error:",
            error
        );

        showError(
            error.message ||
            "Something went wrong while analyzing."
        );

    } finally {

        setLoading(false);

    }

}


// ============================================================
// DISPLAY RESULTS
// ============================================================

function displayResults(data) {

    console.log(
        "Displaying results..."
    );


    if (!resultSection) {

        console.error(
            "resultSection not found."
        );

        return;

    }


    resultSection.classList.remove(
        "hidden"
    );


    // ========================================================
    // RETRIEVED QUESTIONS
    // ========================================================

    const questions =
        data.retrieved_questions ||
        data.questions ||
        data.results ||
        [];


    console.log(
        "Retrieved questions:",
        questions
    );


    // ========================================================
    // QUESTION COUNT
    // ========================================================

    if (questionCount) {

        questionCount.textContent =
            questions.length;

    }


    // ========================================================
    // YEARS
    // ========================================================

    const years = [
        ...new Set(
            questions
                .map(q => q.year)
                .filter(
                    year =>
                        year !== undefined &&
                        year !== null
                )
        )
    ];


    if (yearsFound) {

        yearsFound.textContent =
            years.length > 0
                ? years.join(", ")
                : "-";

    }


    // ========================================================
    // MAIN TOPIC
    // ========================================================

    if (mainTopic) {

        mainTopic.textContent =
            data.query ||
            queryInput.value ||
            "-";

    }


    // ========================================================
    // GEMINI ANALYSIS
    // ========================================================

    let answerText = "";


    // --------------------------------------------------------
    // NEW BACKEND FORMAT
    // --------------------------------------------------------

    if (
        typeof data.analysis === "string" &&
        data.analysis.trim() !== ""
    ) {

        answerText =
            data.analysis;

    }


    // --------------------------------------------------------
    // OLD BACKEND FORMAT
    // --------------------------------------------------------

    else if (
        Array.isArray(data.answer)
    ) {

        const textObject =
            data.answer.find(
                item =>
                    item &&
                    item.type === "text" &&
                    typeof item.text === "string"
            );

        if (textObject) {

            answerText =
                textObject.text;

        }

    }


    else if (
        typeof data.answer === "string"
    ) {

        answerText =
            data.answer;

    }


    else if (
        typeof data.rag_answer === "string"
    ) {

        answerText =
            data.rag_answer;

    }


    // --------------------------------------------------------
    // NO ANSWER
    // --------------------------------------------------------

    if (!answerText) {

        answerText =
            "No AI analysis was returned.";

    }


    // ========================================================
    // SHOW GEMINI ANSWER
    // ========================================================

    if (ragAnswer) {

        ragAnswer.textContent =
            cleanAnswer(answerText);

    }


    // ========================================================
    // SHOW QUESTIONS
    // ========================================================

    displayQuestions(
        questions
    );


    // ========================================================
    // SCROLL TO RESULTS
    // ========================================================

    resultSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });

}


// ============================================================
// DISPLAY QUESTIONS
// ============================================================

function displayQuestions(questions) {

    if (!questionsContainer) {

        console.error(
            "questionsContainer not found."
        );

        return;

    }


    questionsContainer.innerHTML = "";


    if (
        !questions ||
        questions.length === 0
    ) {

        questionsContainer.innerHTML = `
            <div class="no-questions">
                No relevant questions found.
            </div>
        `;

        return;

    }


    questions.forEach(
        (question, index) => {

            const card =
                document.createElement(
                    "div"
                );


            card.className =
                "question-card";


            const year =
                question.year ??
                "Unknown";


            const questionNumber =
                question.question_number ??
                question.number ??
                (index + 1);


            let score = "";


            if (
                question.score !== undefined &&
                question.score !== null
            ) {

                score =
                    Number(
                        question.score
                    ).toFixed(4);

            }


            const questionText =
                question.question ||
                "Question text unavailable.";


            // =================================================
            // OPTIONS
            // =================================================

            let optionsHTML = "";


            if (
                Array.isArray(
                    question.options
                ) &&
                question.options.length > 0
            ) {

                optionsHTML = `
                    <div class="question-options">

                        ${question.options
                            .map(
                                (option, optionIndex) => `
                                    <div class="option">

                                        <strong>
                                            ${String.fromCharCode(
                                                65 +
                                                optionIndex
                                            )}.
                                        </strong>

                                        ${escapeHTML(
                                            String(option)
                                        )}

                                    </div>
                                `
                            )
                            .join("")}

                    </div>
                `;

            }


            // =================================================
            // ANSWER
            // =================================================

            let answerHTML = "";


            if (
                question.answer !== undefined &&
                question.answer !== null &&
                question.answer !== ""
            ) {

                answerHTML = `
                    <div class="question-answer">

                        <strong>
                            Answer:
                        </strong>

                        ${escapeHTML(
                            String(
                                question.answer
                            )
                        )}

                    </div>
                `;

            }


            // =================================================
            // CARD
            // =================================================

            card.innerHTML = `

                <div class="question-header">

                    <span class="question-year">
                        ${escapeHTML(
                            String(year)
                        )}
                    </span>

                    <span class="question-number">
                        Q${escapeHTML(
                            String(
                                questionNumber
                            )
                        )}
                    </span>

                    ${
                        score
                            ? `
                                <span class="similarity-score">
                                    Similarity:
                                    ${score}
                                </span>
                            `
                            : ""
                    }

                </div>


                <div class="question-text">

                    ${escapeHTML(
                        String(
                            questionText
                        )
                    )}

                </div>


                ${optionsHTML}


                ${answerHTML}

            `;


            questionsContainer.appendChild(
                card
            );

        }
    );

}


// ============================================================
// CLEAN GEMINI ANSWER
// ============================================================

function cleanAnswer(text) {

    if (!text) {

        return "";

    }


    let cleaned =
        String(text);


    // Remove accidental code fences

    cleaned =
        cleaned.replace(
            /^```(?:text|markdown)?/i,
            ""
        );


    cleaned =
        cleaned.replace(
            /```$/i,
            ""
        );


    return cleaned.trim();

}


// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHTML(value) {

    const div =
        document.createElement(
            "div"
        );


    div.textContent =
        value;


    return div.innerHTML;

}


// ============================================================
// LOADING
// ============================================================

function setLoading(isLoading) {

    if (loading) {

        if (isLoading) {

            loading.classList.remove(
                "hidden"
            );

        } else {

            loading.classList.add(
                "hidden"
            );

        }

    }


    if (analyzeBtn) {

        analyzeBtn.disabled =
            isLoading;


        if (isLoading) {

            analyzeBtn.textContent =
                "Analyzing...";

        } else {

            analyzeBtn.textContent =
                "Analyze";

        }

    }

}


// ============================================================
// ERROR MESSAGE
// ============================================================

function showError(message) {

    console.error(
        "Frontend error:",
        message
    );


    if (errorMessage) {

        errorMessage.textContent =
            message;

        errorMessage.classList.remove(
            "hidden"
        );

    } else {

        alert(message);

    }

}


// ============================================================
// CLEAR ERROR
// ============================================================

function clearError() {

    if (errorMessage) {

        errorMessage.textContent =
            "";

        errorMessage.classList.add(
            "hidden"
        );

    }

}


// ============================================================
// FORECAST BUTTON
// ============================================================

if (forecastBtn) {

    forecastBtn.addEventListener(
        "click",
        generateForecast
    );

}


// ============================================================
// GENERATE FORECAST
// ============================================================

async function generateForecast() {

    console.log(
        "Forecast button clicked."
    );


    if (forecastLoading) {

        forecastLoading.classList.remove(
            "hidden"
        );

    }


    try {

        const response =
            await fetch(
                `${API_URL}/forecast`
            );


        const data =
            await response.json();


        console.log(
            "Forecast response:",
            data
        );


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Forecast request failed."
            );

        }


        displayForecast(
            data
        );


    } catch (error) {

        console.error(
            "Forecast error:",
            error
        );


        if (forecastContainer) {

            forecastContainer.innerHTML = `
                <div class="error">
                    ${escapeHTML(
                        error.message
                    )}
                </div>
            `;

        }

    } finally {

        if (forecastLoading) {

            forecastLoading.classList.add(
                "hidden"
            );

        }

    }

}


// ============================================================
// DISPLAY FORECAST
// ============================================================

function displayForecast(data) {

    if (!forecastContainer) {

        return;

    }


    const forecasts =
        data.forecasts ||
        [];


    if (
        forecasts.length === 0
    ) {

        forecastContainer.innerHTML = `
            <div>
                No forecast data available.
            </div>
        `;

        return;

    }


    forecastContainer.innerHTML = "";


    forecasts.forEach(
        item => {

            const card =
                document.createElement(
                    "div"
                );


            card.className =
                "forecast-card";


            card.innerHTML = `

                <div class="forecast-topic">

                    ${escapeHTML(
                        String(
                            item.topic
                        )
                    )}

                </div>


                <div class="forecast-value">

                    ${escapeHTML(
                        String(
                            item.forecast_count
                        )
                    )}

                </div>


                <div class="forecast-year">

                    Forecast for
                    ${escapeHTML(
                        String(
                            item.forecast_year
                        )
                    )}

                </div>


                <div class="forecast-trend">

                    Trend:
                    ${escapeHTML(
                        String(
                            item.trend
                        )
                    )}

                </div>

            `;


            forecastContainer.appendChild(
                card
            );

        }
    );

}


// ============================================================
// END OF SCRIPT
// ============================================================