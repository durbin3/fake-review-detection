import numpy as np
import matplotlib.pyplot as plt
def plot_results():
    x = [.001,.01,.1,1,10,100]
    y_100 = [.500968,.67811424,.8513070,.912489243,.95296364,.9570783]
    y_80 = [.4977644,.648065,.8452953,.91084815,.9571049,.963794668]
    y_60 = [.501479,.6428507,.83541013,.90793366,.96539668,.9744509]
    y_40 = [.5028909,.6364125,.8180718,.90278338,.9751916095,.9858814]
    y_20 = [.500874008,.49912599,.770606,.8900094,.9893774,.99825198]
    yvals = [y_20,y_40,y_60,y_80,y_100]
    
    p = ['20%','40%','60%','80%','100%']
    for i in range(len(p)):
        y = yvals[i]
        px = p[i]
        plot(x,y,px)
        
        
        
def plot(x,y,c):
    plt.clf()
    plt.yticks(np.arange(5), ['20%','40%','60%','80%','100%'])
    plt.ylabel("Training Accuracy")
    plt.xlabel('C value')
    plt.title("Training Accuracy vs C value using "+ c+ " of Data")
    plt.scatter(x,y)
    plt.plot(x,y)
    plt.savefig("c_plot_"+c[:len(c)-1])
    
    
plot_results()