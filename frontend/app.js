import React, { useState } from "react";

function App() {
  const [inputText, setInputText] = useState("");
  const [outputText, setOutputText] = useState("");

  const handleTranslate = async () => {
    // Fake output since we won't actually run Flask here
    setOutputText("अनुवादित पाठ (demo output)");
  };

  return (
    <div style={{ margin: "50px", fontFamily: "Arial" }}>
      <h1>English → Hindi Translator</h1>
      <textarea
        rows="4"
        cols="50"
        placeholder="Enter English text here"
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
      />
      <br />
      <button onClick={handleTranslate}>Translate</button>
      <h3>Hindi Translation:</h3>
      <p>{outputText}</p>
    </div>
  );
}

export default App;
