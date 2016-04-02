package ml.arunreddy.tl.data.sentiment;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by arun on 3/28/16.
 */
public class MultiDomainSentimenContentHandler extends DefaultHandler {

    private StringBuilder reviewTextBuilder = new StringBuilder();
    private StringBuilder uniqueIdBuilder = new StringBuilder();
    private StringBuilder reviewerBuilder = new StringBuilder();
    private StringBuilder ratingBuilder  = new StringBuilder();

    private List<String> reviewTextList = new ArrayList<>();
    private List<String> reviewerList = new ArrayList<>();
    private List<String> uniqueIdList = new ArrayList<>();
    private List<String> ratingList = new ArrayList<>();

    private boolean textFlag = false;
    private boolean uniqueFlag = false;
    private boolean reviewerFlag = false;
    private boolean ratingFlag = false;


    public List<String> getReviewTextList() {
        return reviewTextList;
    }

    public List<String> getReviewerList() {
        return reviewerList;
    }

    public List<String> getUniqueIdList() {
        return uniqueIdList;
    }

    public List<String> getRatingList() { return ratingList;}

    @Override
    public void startElement(String uri, String localName, String qName, Attributes attributes) throws SAXException {
        super.startElement(uri, localName, qName, attributes);

        switch (qName) {
            case "review":
                break;
            case "reviewer":
                reviewerFlag = true;
                break;
            case "review_text":
                textFlag = true;
                break;
            case "unique_id":
                uniqueFlag = true;
                break;
            case "rating":
                ratingFlag = true;
                break;
            default:

        }

    }

    @Override
    public void endElement(String uri, String localName, String qName) throws SAXException {
        super.endElement(uri, localName, qName);

        switch (qName) {
            case "review":
                String reviewText = reviewTextBuilder.toString().replaceAll("\n", "").trim();
                String uniqueId = uniqueIdBuilder.toString().replaceAll("\n", "").trim();
                String reviewer = reviewerBuilder.toString().replaceAll("\n", "").trim();
                String rating = ratingBuilder.toString().replaceAll("\n", "").trim();


                reviewerList.add(reviewer);
                uniqueIdList.add(uniqueId);
                reviewTextList.add(reviewText);
                ratingList.add(rating);

                reviewTextBuilder.setLength(0);
                uniqueIdBuilder.setLength(0);
                reviewerBuilder.setLength(0);
                ratingBuilder.setLength(0);
                break;
            case "review_text":
                textFlag = false;
                break;
            case "unique_id":
                uniqueFlag = false;
                break;
            case "reviewer":
                reviewerFlag = false;
                break;
            case "rating":
                ratingFlag = false;
                break;

            default:

        }
    }

    @Override
    public void characters(char[] ch, int start, int length) throws SAXException {
        super.characters(ch, start, length);
        if (textFlag) {
            reviewTextBuilder.append(new String(ch, start, length));
        }
        if (uniqueFlag) {
            uniqueIdBuilder.append(new String(ch, start, length));
        }
        if (reviewerFlag) {
            reviewerBuilder.append(new String(ch, start, length));
        }

        if(ratingFlag){
            ratingBuilder.append(new String(ch, start, length));
        }
    }

    @Override
    public void ignorableWhitespace(char[] ch, int start, int length) throws SAXException {
        super.ignorableWhitespace(ch, start, length);
    }
}

