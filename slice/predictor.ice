module MyPredictor {
    interface Predictor {
        string sendTrainingData(string csvData);
        string trainModel();
        string predict(string inputData);
    };
};
