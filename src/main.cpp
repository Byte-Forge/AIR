#include <SFML/Window.hpp>
#include "NeuralNetwork.hpp"
#include "NeuralNetworkTrainer.hpp"
#include "DataReader.hpp"
#include <memory>


// Idee: speichere für jeden zug die aktuelle situation und die ausgeführte aktion
// -> wenn der spieler gewinnt ist das datenset gültig und wird an die datenbank angehängt falls nicht lösche die daten

using namespace air;

int main()
{
    sf::Window window(sf::VideoMode(800, 600), "AI Research");

	////seed random number generator
	srand((unsigned int)time(0)); 

	////create data set reader and load data file
	DataReader d;
	d.loadDataFile("../../src/data.csv", 16, 3);
	d.setCreationApproach(STATIC, 10);

	////create neural network
	std::shared_ptr<NeuralNetwork> nn = std::make_shared<NeuralNetwork>(16, 20, 3, 3);

	//create neural network trainer
	NeuralNetworkTrainer nT(nn);
	nT.setTrainingParameters(0.001, 0.9, false);
	nT.setStoppingConditions(200, 90);
	nT.enableLogging("log.csv", 5);

	//train neural network on data sets
	for (int i = 0; i < d.getNumTrainingSets(); i++)
	{
		nT.trainNetwork(d.getTrainingDataSet());
	}

	//save the weights
	nn->saveWeights("weights.csv");

    // run the program as long as the window is open
    while (window.isOpen())
    {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }
    }

    return 0;
}