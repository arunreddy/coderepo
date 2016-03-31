# Read Lines
print ("Stanford Sentiment Parser")

for domain in ["kitchen","dvd","electronics","books"]:
    for sentiment in ["negative","positive"]:
        fname = "{0}_{1}.stanford.labels".format(domain,sentiment)
        lines = tuple(open(fname,'r'));
        count=0;
        total=0;
        for line in lines:
            scoreStr=line.rstrip('\n').split("\t")[1]
            score=float(scoreStr);
            total=total+1;
            if 'positive' in fname:
                if score >= 0.5:
                    count=count+1
            else: 
                if score < 0.5:
                    count=count+1

        accuracy=(count*100)/total
        print("{0}\t{1}\t{2}\t{3}\t{4}%".format(domain,sentiment,count,total,accuracy))
