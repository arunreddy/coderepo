/*
 * The MIT License (MIT)
 * Copyright (c) 2016 Arun Reddy Nelakurthi
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

package ml.arunreddy.tl.classifier.sentiment;

import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.trees.Tree;
import ml.arunreddy.tl.data.sentiment.MultiDomainSentimentDataParser;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/**
 * Created by arun on 3/28/16.
 */
public class MultiDomainStanfordSentimentClassifier {

    public double getSentimentScore(String text){
        Properties props;
        props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, parse, sentiment");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        int mainSentiment = 0;
        if (text != null && !text.isEmpty()) {
            int longest = 0;
            Annotation annotation = pipeline.process(text);
            for (CoreMap sentence : annotation.get(CoreAnnotations.SentencesAnnotation.class)) {
                Tree tree = sentence.get(SentimentCoreAnnotations.AnnotatedTree.class);
                int sentiment = RNNCoreAnnotations.getPredictedClass(tree);
                String partText = sentence.toString();
                if (partText.length() > longest) {
                    mainSentiment = sentiment;
                    longest = partText.length();
                }

            }
        }

        assert mainSentiment >= 0;
        assert mainSentiment <= 4;

        double value = (double)mainSentiment/4.0;

        return value;
    }


    public static void main(String args[]) throws IOException{

        if(args.length!=5){
            System.out.println("Error.. Need 4 Arguments.");
            System.out.println("Usage args: <data path> <domain> <sentiment type> <start> <len>");
            System.exit(0);
        }

        String path=args[0];
        String domain = args[1];
        String sentimentType = args[2];
        int start = Integer.parseInt(args[3]);
        int len = Integer.parseInt(args[4]);


        MultiDomainStanfordSentimentClassifier classifier = new MultiDomainStanfordSentimentClassifier();
        String filePath = path+"/"+domain+"/"+sentimentType+".review.text";
        String outDir = path+"/outdir";

        List<String> allLines= Files.readAllLines(Paths.get(filePath));
        List<String> subset = allLines.subList(start,start+len);

        List<String> reviewScores = new ArrayList<>();
        for(String reviewText:subset){
            String[] strings=reviewText.split("\t");

            double sentimentScore = classifier.getSentimentScore(strings[2]);
            reviewScores.add(strings[0]+"\t"+sentimentScore);
        }

        StringBuilder reviewScoresBuilder=new StringBuilder();
        for (String reviewScore:reviewScores){
            reviewScoresBuilder.append(reviewScore+"\n");
        }

        FileWriter writer = new FileWriter(outDir+"/"+domain+"_"+sentimentType+"_"+String.format("%04d",start)+".stanford");
        writer.write(reviewScoresBuilder.toString());
        writer.flush();
        writer.close();

    }
}
