def change_label(filename,localname):
    datas = []
    with open(filename,'r') as f1:
        for content in f1:
            text,label = content.split('\t')

            datas.append((text,label))
    with open(localname,'w') as f1:
        for text,label in datas:
            f1.write(text)
            f1.write('[SEP]')
            f1.write(label)
            f1.write('\n')

change_label('./dev.txt','./dev.txt')
change_label('./test.txt','./test.txt')
change_label('./train.txt','./train.txt')

