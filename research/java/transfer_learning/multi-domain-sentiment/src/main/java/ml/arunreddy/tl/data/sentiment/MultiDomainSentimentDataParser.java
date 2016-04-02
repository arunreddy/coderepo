package ml.arunreddy.tl.data.sentiment;

import org.xml.sax.InputSource;
import org.xml.sax.SAXException;
import org.xml.sax.XMLReader;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;

/**
 * Created by arun on 3/25/16.
 */
public class MultiDomainSentimentDataParser {

    public static void main(String args[]) throws Exception {

        MultiDomainSentimentDataParser parser = new MultiDomainSentimentDataParser();

        String[] folders = {"dvd", "electronics", "books", "kitchen_&_housewares"};
        String[] files = {"positive.review", "negative.review"};
        for (String folder : folders) {
            for (String file : files) {
                String filePath = "/tmp/sorted_data/" + folder + "/" + file;
                System.out.println(filePath);
                MultiDomainSentimenContentHandler contentHandler = parser.parse(filePath);
                List<String> reviewTextList = contentHandler.getReviewTextList();
                List<String> reviewerList = contentHandler.getReviewerList();
                List<String> uniqueIdList = contentHandler.getUniqueIdList();
                List<String> ratingList = contentHandler.getRatingList();

                StringBuilder textBuilder = new StringBuilder();
                StringBuilder userInfoBuilder = new StringBuilder();

                for (int i = 0; i < reviewerList.size(); i++) {
                    textBuilder.append(folder+"_"+file+"_"+i+"\t"+ratingList.get(i)+"\t"+reviewTextList.get(i) + "\n");
                }

                for (int i = 0; i < reviewerList.size(); i++) {
                    userInfoBuilder.append(folder+"_"+file+"_"+i+"\t"+uniqueIdList.get(i) + "\t" + reviewerList.get(i) + "\n");
                }


                FileWriter reviewTextWriter = new FileWriter(filePath + ".text");
                FileWriter userInfoWriter = new FileWriter(filePath + ".info");


                reviewTextWriter.write(textBuilder.toString());
                userInfoWriter.write(userInfoBuilder.toString());

                reviewTextWriter.flush();
                reviewTextWriter.close();

                userInfoWriter.flush();
                userInfoWriter.close();
            }
        }
    }

    public String xmlEscapeText(String t) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < t.length(); i++) {
            char c = t.charAt(i);
            switch (c) {
//                case '<': sb.append("&lt;"); break;
//                case '>': sb.append("&gt;"); break;
                case '\"':
                    sb.append("&quot;");
                    break;
                case '&':
                    sb.append("&amp;");
                    break;
                case '\'':
                    sb.append("&apos;");
                    break;
                case 0x1a:
                    sb.append("");
                    break;
                default:
                    if (c > 0x7e) {
                        sb.append("&#" + ((int) c) + ";");
                    } else
                        sb.append(c);
            }
        }
        return sb.toString();
    }

    public MultiDomainSentimenContentHandler parse(String filePath) {
        try (FileReader fr = new FileReader(filePath)) {

            byte[] encoded = Files.readAllBytes(Paths.get(filePath));
            String fileContent = new String(encoded);

            // Clean up the string.

            // Remove illegal characters from XML.
            // XML 1.1
            // [#x1-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]
            String xml11pattern = "[^"
                    + "\u0001-\uD7FF"
                    + "\uE000-\uFFFD"
                    + "\ud800\udc00-\udbff\udfff"
                    + "]+";
            fileContent = fileContent.replaceAll(xml11pattern, "");

            // escape non xml safe characters.
            fileContent = xmlEscapeText(fileContent);
            fileContent = fileContent.replaceAll("\"", "");
            fileContent = fileContent.replaceAll("\'", "");


            // Add root element.
            fileContent = "<root>" + fileContent + "</root>";

            StringReader stringReader = new StringReader(fileContent);
            SAXParserFactory spf = SAXParserFactory.newInstance();
            spf.setNamespaceAware(true);
            SAXParser saxParser = spf.newSAXParser();
            XMLReader xmlReader = saxParser.getXMLReader();
            MultiDomainSentimenContentHandler contentHandler = new MultiDomainSentimenContentHandler();
            xmlReader.setContentHandler(contentHandler);
            xmlReader.parse(new InputSource(stringReader));

            return contentHandler;

        } catch (FileNotFoundException ex) {
            ex.printStackTrace();
        } catch (IOException ex) {
            ex.printStackTrace();
        } catch (ParserConfigurationException e) {
            e.printStackTrace();
        } catch (SAXException e) {
            e.printStackTrace();
        }

        return null;
    }


}
