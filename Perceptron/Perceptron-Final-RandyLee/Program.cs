using System;

namespace Perceptron_Final_RandyLee
{
    class ActivationProgram
    {
        static void Main(string[] args)
        {
            try
            {
                Console.WriteLine("Begin neural network activation sigmoid function perceptron.");
     
                DummyNeuralNetwork dnn = new DummyNeuralNetwork();

                Console.WriteLine("\nGrabbing User Input...");
                Console.Write("Input 1: ");
                int input1 = Convert.ToInt32(Console.ReadLine());
                Console.Write("Input 2: ");
                int input2 = Convert.ToInt32(Console.ReadLine());

                // create new double array with user inputs
                double[] inputs = new double[] { input1, input2 };
                dnn.SetInputs(inputs);

                // set weights, we only need two inputs.
                decimal[] randomWeights = new decimal[12];

                Console.WriteLine("\nCalculating Random Weights: ");
                Random randomNum = new Random();
                for (int i = 0; i < randomWeights.Length; i++)
                {
                    randomWeights[i] = (decimal)randomNum.NextDouble();
                    Console.WriteLine(randomWeights[i]);
                }


                dnn.SetWeights(randomWeights);

                // with these weights and biases, we can use log-sigmoid activation

                Console.WriteLine("\nComputing outputs using Log-Sigmoid activation");
                dnn.ComputeOutputs("logsigmoid");
                Console.WriteLine("\nLog-Sigmoid NN outputs are: ");
                Console.Write(dnn.outputs[0].ToString("F4") + " ");
                Console.WriteLine(dnn.outputs[1].ToString("F4"));

                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                Console.ReadLine();
            }
        }
    } 


    public class DummyNeuralNetwork
    {
        public double[] inputs;

        public double ihWeight00;
        public double ihWeight01;
        public double ihWeight10;
        public double ihWeight11;
        public double ihBias0;
        public double ihBias1;

        /// <summary>
        ///  ihSum is the sum of the products of inputs, weights, plus bias value 
        ///  howeights holds the weight for nodes 1 to output node 0
        ///  ih Result() is the value emitted from hidden node 0 after an activation function has been applied to our ihSum value
        /// </summary>

        public double ihSum0;
        public double ihSum1;
        public double ihResult0;
        public double ihResult1;

        public double hoWeight00;
        public double hoWeight01;
        public double hoWeight10;
        public double hoWeight11;
        public double hoBias0;
        public double hoBias1;

        public double hoSum0;
        public double hoSum1;
        public double hoResult0;
        public double hoResult1;

        public double[] outputs;

        public DummyNeuralNetwork()
        {
            this.inputs = new double[2];
            this.outputs = new double[2];
        }

        public void SetInputs(double[] inputs)
        {
            inputs.CopyTo(this.inputs, 0);
        }

        public void SetWeights(decimal[] weightsAndBiases)
        {
            int k = 0;
            ihWeight00 = (double)weightsAndBiases[k++];
            ihWeight01 = (double)weightsAndBiases[k++];
            ihWeight10 = (double)weightsAndBiases[k++];
            ihWeight11 = (double)weightsAndBiases[k++];
            ihBias0 = (double)weightsAndBiases[k++];
            ihBias1 = (double)weightsAndBiases[k++];

            hoWeight00 = (double)weightsAndBiases[k++];
            hoWeight01 = (double)weightsAndBiases[k++];
            hoWeight10 = (double)weightsAndBiases[k++];
            hoWeight11 = (double)weightsAndBiases[k++];
            hoBias0 = (double)weightsAndBiases[k++];
            hoBias1 = (double)weightsAndBiases[k++];
        }

        public void ComputeOutputs(string activationType)
        {
            // Assumes this.inputs[] have values
            // ihSum- is the sum of the products of inputs and weights, plus the bias value for node 0 before activation has been applied
            ihSum0 = (inputs[0] * ihWeight00) + (inputs[1] * ihWeight10);
            ihSum0 += ihBias0;
            ihSum1 = (inputs[0] * ihWeight01) + (inputs[1] * ihWeight11);
            ihSum1 += ihBias1;
            ihResult0 = Activation(ihSum0, activationType, "ih");
            ihResult1 = Activation(ihSum1, activationType, "ih");

            hoSum0 = (ihResult0 * hoWeight00) + (ihResult1 * hoWeight10);
            hoSum0 += hoBias0;
            hoSum1 = (ihResult0 * hoWeight01) + (ihResult1 * hoWeight11);
            hoSum1 += hoBias1;
            hoResult0 = Activation(hoSum0, activationType, "ho");
            hoResult1 = Activation(hoSum1, activationType, "ho");

            outputs[0] = hoResult0;
            outputs[1] = hoResult1;
        }

        public double Activation(double x, string activationType, string layer)
        {
            if (activationType == "logsigmoid")
                return LogSigmoid(x);

            // we could add softmax and hyperbolic tangent too here, I was playing around with it
            // todo: add hyperbolic tangent and softmax based on activation type passed to method
            else
                throw new Exception("Not implemented");
        }

        // remember that a sigmoid function is 1 / ( 1 + e^x )
        public double LogSigmoid(double x)
        {
            if (x < -45.0) return 0.0;
            else if (x > 45.0) return 1.0;
            else return 1.0 / (1.0 + Math.Exp(-x));
     
        }
    }
}