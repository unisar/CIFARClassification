import org.apache.commons.math3.optim.nonlinear.vector.Weight;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.ComposableRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URI;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static java.lang.System.exit;

/**
 * Created by unisar 
 */
public class AllConvolutionDL4J {
    private static final Logger log = LoggerFactory.getLogger(AllConvolutionDL4J.class);
    private static String X_Train;
    private static String Y_Train;
    private static String X_Test;
    private static String Image_Path;

    public static void main(String[] args) throws Exception {
        if (args.length < 4)
        {
            System.out.println("Usage: X_Train Y_Train X_Test IMAGE_FOLDER");
            exit(1);
        }

        X_Train = args[0];
        Y_Train = args[1];
        X_Test = args[2];
        Image_Path = args[3];

        int nChannels = 3;
        int outputNum = 10;
        int batchSize = 100;
        int nEpochs = 50;
        int iterations = 1;
        int seed = 123;

        Nd4j.ENFORCE_NUMERICAL_STABILITY = true;


        RecordReader recordReader = loadTrainingData();
        log.info("Load data....");

        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader, batchSize, 1, 10);
//        DataSetIterator testSetIterator = new RecordReaderDataSetIterator(loadTestingData(), 1);


        log.info("Build model....");
        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed).miniBatch(true)
                .iterations(iterations)
                .regularization(true).l2(0.0005)
                .learningRate(0.1) //.biasLearningRate(0.2)
                .learningRateDecayPolicy(LearningRatePolicy.Step).lrPolicyDecayRate(0.1).lrPolicySteps(45000/batchSize * 50)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new ConvolutionLayer.Builder().kernelSize(new int[] { 3, 3 }).stride(new int[] { 1, 1 }).padding(new int[] { 1, 1 })
                    .nOut(96)
                    .activation("relu")
                    .build())
                .layer(1, new ConvolutionLayer.Builder().kernelSize(new int[] { 3, 3 }).stride(new int[] { 1, 1 }).padding(new int[] { 1, 1 })
                    .nOut(96)
                    .activation("relu")
                    .build())
                .layer(2, new ConvolutionLayer.Builder().kernelSize(new int[] {2, 2}).stride(new int[] {2, 2})
                    .nOut(96).activation("relu")
                    .dropOut(0.5)
                    .build())
                .layer(3, new ConvolutionLayer.Builder().kernelSize(new int[] { 3, 3 }).stride(new int[] { 1, 1 }).padding(new int[] { 1, 1 })
                    .nOut(192)
                    .activation("relu")
                    .build())
                .layer(4, new ConvolutionLayer.Builder().kernelSize(new int[] { 3, 3 }).stride(new int[] { 1, 1 }).padding(new int[] { 1, 1 })
                    .nOut(192)
                    .activation("relu")
                    .build())
                .layer(5, new ConvolutionLayer.Builder().kernelSize(new int[] {2, 2}).stride(new int[] {2, 2})
                    .nOut(192).activation("relu")
                    .dropOut(0.5)
                    .build())
                .layer(6, new ConvolutionLayer.Builder().kernelSize(new int[] { 3, 3 }).stride(new int[] { 1, 1 }).padding(new int[] { 1, 1 })
                    .nOut(192)
                    .activation("relu")
                    .build())
                .layer(7, new ConvolutionLayer.Builder().kernelSize(new int[] {1,1}).stride(new int[] { 1, 1 })
                    .nOut(192)
                    .activation("relu")
                    .build())
                .layer(8, new ConvolutionLayer.Builder().kernelSize(new int[] {1,1}).stride(new int[] { 1, 1 })
                    .nOut(10)
                    .activation("relu")
                    .build())
                .layer(9, new SubsamplingLayer.Builder().kernelSize(new int[] {8, 8}).stride(new int[] { 1, 1 }).build())
                .layer(10, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nOut(outputNum)
                    .activation("softmax")
                    .build()).setInputType(InputType.convolutionalFlat(32, 32, 3))
            .backprop(true)
            .pretrain(false);

        MultiLayerConfiguration conf = builder.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        log.info("Train model....");
        model.setListeners(new IterationListener[] {new ScoreIterationListener(1), new HistogramIterationListener(1)});
        DataSet cifarDataSet;
        SplitTestAndTrain trainAndTest;

        for( int i=0; i<nEpochs; i++ ) {
            dataSetIterator.reset();
            List<INDArray> testInput = new ArrayList<>();
            List<INDArray> testLabels = new ArrayList<>();

            while (dataSetIterator.hasNext())
            {
                cifarDataSet = dataSetIterator.next();
                trainAndTest = cifarDataSet.splitTestAndTrain((int)(batchSize*0.9));
                DataSet trainInput = trainAndTest.getTrain();
                testInput.add(trainAndTest.getTest().getFeatureMatrix());
                testLabels.add(trainAndTest.getTest().getLabels());
                model.fit(trainInput);
            }
            File file = File.createTempFile("Epoch", String.valueOf(i));

            ModelSerializer.writeModel(model, file, true );

            log.info("*** Completed epoch {} ***", i);

            log.info("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            for (int j = 0; j < testInput.size(); j++) {
                INDArray output = model.output(testInput.get(j));
                eval.eval(testLabels.get(i), output);
            }
            log.info(eval.stats());

//            testSetIterator.reset();
            dataSetIterator.reset();
        }
        log.info("****************Example finished********************");
    }


    public static RecordReader loadTestingData() throws Exception {
        RecordReader imageReader = new ImageRecordReader(32, 32, 3, 255);
        List<URI> images = new ArrayList<>();
        for (String number: Files.readAllLines(Paths.get(X_Test), Charset.defaultCharset()))
            images.add(URI.create("file:/"+ Image_Path + "/Test/" + number + ".png"));

        InputSplit splits = new CollectionInputSplit(images);
        imageReader.initialize(splits);


        return imageReader;
    }

    public static RecordReader loadTrainingData() throws Exception {
        RecordReader imageReader = new ImageRecordReader(32, 32, 3, 255);
        List<URI> images = new ArrayList<>();
        for (String number: Files.readAllLines(Paths.get(X_Train), Charset.defaultCharset()))
            images.add(URI.create("file:/"+ Image_Path + "/Train/" + number + ".png"));

        InputSplit splits = new CollectionInputSplit(images);
        imageReader.initialize(splits);

        RecordReader labelsReader = new CSVRecordReader();
        labelsReader.initialize(new FileSplit(new File(Y_Train)));

        return new ComposableRecordReader(imageReader, labelsReader);
    }
}
