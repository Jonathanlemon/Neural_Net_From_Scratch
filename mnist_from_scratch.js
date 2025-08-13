const math = require('mathjs');
const mnist = require('mnist');

let weights1, biases1, weights2, biases2, weights3, biases3;
let learningRate = 0.05;
const inputSize = 784;
const hiddenSize = 128;   // hidden layer
const hiddenSize2 = 64;   // second hidden layer
const outputSize = 10;    // digits 0–9

function init(){
    const { training, test } = mnist.set(10000, 2000);

    // Save data globally
    global.trainingData = (training);
    global.testData = (test);

    function randomMatrix(rows, cols, min = 0, max = 1) {
      return math.matrix(
        Array.from({ length: rows }, () =>
          Array.from({ length: cols }, () => Math.random() * (max - min) + min)
        )
      );
    }

    // Initialize weights and biases with small random values
    // Layer 1: input -> hidden1
    weights1 = randomMatrix(hiddenSize, inputSize, -0.5, 0.5);
    // Layer 2: hidden1 -> hidden2
    weights2 = randomMatrix(hiddenSize2, hiddenSize, -0.5, 0.5);
    // Layer 3: hidden2 -> output
    weights3 = randomMatrix(outputSize, hiddenSize2, -0.5, 0.5);
    biases1 = math.zeros([hiddenSize, 1]);
    biases2 = math.zeros([hiddenSize2, 1]);
    biases3 = math.zeros([outputSize, 1]);

    console.log("Initialized weights and biases.");
    logStats("Weights1:", weights1);
    logStats("Weights2:", weights2);
    logStats("Weights3:", weights3);
}

function relu(x) { return math.map(x, v => Math.max(0, v)); }
function reluDerivative(x) { return math.map(x, v => v > 0 ? 1 : 0); }

function softmax(x) {
    const maxVal = math.max(x); // for numerical stability
    const shifted = math.subtract(x, maxVal); // subtract max from each element

    const exps = math.map(shifted, math.exp); // apply exp element-wise
    const sumExp = math.sum(exps);

    return math.divide(exps, sumExp); // element-wise divide
}

function logStats(name, matrix) {
  const arr = math.flatten(matrix).valueOf();
  const min = Math.min(...arr);
  const max = Math.max(...arr);
  const avg = arr.reduce((sum, v) => sum + v, 0) / arr.length;
  console.log(`${name} - min: ${min.toFixed(5)}, max: ${max.toFixed(5)}, avg: ${avg.toFixed(5)}`);
}

function calculateLoss(dataset) {
    let totalLoss = 0;
    const epsilon = 1e-12;

    for (let i = 0; i < dataset.length; i++) {
        const X = math.matrix(dataset[i].input).reshape([inputSize, 1]);
        const Y = math.matrix(dataset[i].output).reshape([outputSize, 1]);

        const { a3 } = forward_prop(X);
        const logProbs = math.map(a3, v => Math.log(v + epsilon));
        const loss = -math.sum(math.dotMultiply(Y, logProbs));

        totalLoss += loss;
    }

    return totalLoss / dataset.length;
}


function forward_prop(input){
    input = math.resize(input, [inputSize, 1]);
    //console.log(logStats("input_", input));
    let z1 = math.add(math.multiply(weights1, input), biases1);
    //logStats("z1", z1); 

    let a1 = relu(z1);
    //logStats("a1", a1);

    let z2 = math.add(math.multiply(weights2, a1), biases2);
    //logStats("z2", z2);

    let a2 = relu(z2);
    //logStats("a2", a2);

    let z3 = math.add(math.multiply(weights3, a2), biases3);
    //logStats("z3", z3);

    let a3 = softmax(z3);
    // Softmax output is probability, so min/max between 0 and 1 usually, average ~0.1
    //logStats("a3", a3);

  return { z1, a1, z2, a2, z3, a3 };
}

function shuffle(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

function back_prop(x, y, result){

    x = math.reshape(x, [inputSize, 1]);
    y = math.reshape(y, [outputSize, 1]);
    //should generate one gradient vector for example m. Calculate the derivatives and solve for the values for that input. Will be summed elsewhere and then averaged to find the average value of derivative for each parameter
    //SOLVING FOR: dW3, dW2, dW1, and dB3, dB2, dB1. Get the accurate expressions, and then plug in values to get numeric answers as a gradient vector.
    let dz3, dz2, dz1, dw3, dw2, dw1, db3, db2, db1;
    //dC/dz3
    dz3 = math.subtract(result.a3, y); //This is a simplified way, assuming softmax activation on the last layer, and then cross-entry for the loss function. This derivative is already solved, and basically is a clean way to already have a partial derivative for the pre-activated last layer output to the loss. Makes things easier
    //solving for dw3. dC/dw3 = dz3/dw3 * dC/dz3
    dw3 = math.multiply(dz3,math.transpose(result.a2)); // Should produce an output with the same shape as the weights, so each entry corresponds to one particular weight's partial derivative toward Cost
    //db3. dC/db3 = dz3/db3 * dC/dz3
    db3 = dz3; //This is a constant, because it derives to dz3/db3 = 1 * w*a, which simplifies to a constant 1.

    dz2 = math.dotMultiply(math.multiply(math.transpose(weights3), dz3), reluDerivative(result.z2)); // This is the nifty chain rule, basically for each node in l2, changing it changes every node in l3. Changing an l2 node slightly, changes the activated output by derivative of relu, and that chains to, changes each node in l3 by its corresponding weight, and that change further contributes to the overall Cost change by that L3's node derivative. So basically we transpose the weight matrix, so that the matrix dot product, sums every weight from l2*its corresponding l3 node derivative. So, z2 changes C by z2's effect on A2, * A2's effect on Z3 (which is all the weights times each z3's derivative), * z3's effect on C.
    dw2 = math.multiply(dz2,math.transpose(result.a1));
    db2 = dz2;

    dz1 = math.dotMultiply(math.multiply(math.transpose(weights2), dz2), reluDerivative(result.z1));
    dw1 = math.multiply(dz1,math.transpose(x));
    db1 = dz1;

    return { dw1, db1, dw2, db2, dw3, db3 };
}


// Pre-allocate gradient accumulators outside the epochs (reuse)
let dw1_sum, db1_sum, dw2_sum, db2_sum, dw3_sum, db3_sum;

function resetGradients() {
  dw1_sum = math.zeros(math.size(weights1));
  db1_sum = math.zeros(math.size(biases1));
  dw2_sum = math.zeros(math.size(weights2));
  db2_sum = math.zeros(math.size(biases2));
  dw3_sum = math.zeros(math.size(weights3));
  db3_sum = math.zeros(math.size(biases3));
}

function learn(epochs){
  let batchSize = 64;

  for(let e=0;e<epochs;e++){
    shuffle(trainingData);

    resetGradients(); // zero accumulators once per epoch start

    let iterations = 0;

    for(let i=0;i<trainingData.length;i++){
      iterations++;

      // Use already-matrixed inputs and outputs, no conversion here
      let result = forward_prop(trainingData[i].input);
      let gradient = back_prop(trainingData[i].input, trainingData[i].output, result);

      // Accumulate gradients
      dw1_sum = math.add(dw1_sum, gradient.dw1);
      db1_sum = math.add(db1_sum, gradient.db1);
      dw2_sum = math.add(dw2_sum, gradient.dw2);
      db2_sum = math.add(db2_sum, gradient.db2);
      dw3_sum = math.add(dw3_sum, gradient.dw3);
      db3_sum = math.add(db3_sum, gradient.db3);

      // Apply update every batchSize examples or at the last batch
      if(iterations === batchSize || i === trainingData.length -1){
        // Average gradients over batch
        dw1_sum = math.divide(dw1_sum, iterations);
        db1_sum = math.divide(db1_sum, iterations);
        dw2_sum = math.divide(dw2_sum, iterations);
        db2_sum = math.divide(db2_sum, iterations);
        dw3_sum = math.divide(dw3_sum, iterations);
        db3_sum = math.divide(db3_sum, iterations);

        if(i === trainingData.length-1){
            logStats("dw1: ", dw1_sum);
            logStats("db1: ", db1_sum);
            logStats("dw2: ", dw2_sum);
            logStats("db2: ", db2_sum);
            logStats("dw3: ", dw3_sum);
            logStats("db3: ", db3_sum);
        }


        // Gradient descent step
        weights1 = math.subtract(weights1, math.multiply(dw1_sum, learningRate));
        biases1 = math.subtract(biases1, math.multiply(db1_sum, learningRate));
        weights2 = math.subtract(weights2, math.multiply(dw2_sum, learningRate));
        biases2 = math.subtract(biases2, math.multiply(db2_sum, learningRate));
        weights3 = math.subtract(weights3, math.multiply(dw3_sum, learningRate));
        biases3 = math.subtract(biases3, math.multiply(db3_sum, learningRate));

        // Reset accumulators for next batch
        resetGradients();   
        iterations = 0;
      }
    }
    console.log("Epoch:", e, "completed.");
    const trainLoss = calculateLoss(trainingData);
    console.log(`Epoch ${e} - Train Loss: ${trainLoss.toFixed(6)}`);
    learningRate = learningRate * 0.95;
    logStats("weights1:", weights1);
    logStats("biases1:", biases1);
    logStats("weights2:", weights2);
    logStats("biases2:", biases2);
    logStats("weights3:", weights3);
    logStats("biases3:", biases3);
    make_prediction(0, 1);
  }
}



function make_prediction(verbose, use_training){
    let correct_guesses = 0;

    let predictionData = testData;
    if(use_training == 1){
        predictionData = trainingData;
    }
    let total = predictionData.length;
    //Use the model to make prediction across test data and get results/accuracy/statistics
    for(let i=0;i<predictionData.length;i++){
        const inputVec = math.matrix(predictionData[i].input);
        if (!predictionData[i].input || predictionData[i].input.includes(undefined)) {
            console.warn("Bad input at index", i);
            continue;
        }
        else{
            const result = forward_prop(inputVec);
            let prediction = result.a3.toArray().flat().indexOf(math.max(result.a3)); // index of highest value = predicted digit
            let correct = predictionData[i].output.indexOf(math.max(math.matrix(predictionData[i].output)));
            if(verbose == 1){
                console.log("Predicting: "+prediction+" with "+result.a3, " from a2: ",result.a2, " vs actual ",correct);
            }
            if(prediction == correct){
                correct_guesses++;
                //console.log("Nice!");
            }
            
        }
        
    }

    console.log(correct_guesses + " out of " + total + " predictions correct. "+(correct_guesses/total)+" accuracy value.")
}

init();
//let result = forward_prop(trainingData[2].input);
make_prediction(1, 0);
learn(30);