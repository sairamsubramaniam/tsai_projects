Make a DNN such that:

    it's first block uses following code:
        import datetime from datetime
        print("Current Date/Time: ", datetime.now())
    uses the Modules you have written
    calls following DNN from a file called QuizDNN.py:
        x1 = Input
        x2 = Conv(x1)
        x3 = Conv(x1 + x2)
        x4 = MaxPooling(x1 + x2 + x3)
        x5 = Conv(x4)
        x6 = Conv(x4 + x5)
        x7 = Conv(x4 + x5 + x6)
        x8 = MaxPooling(x5 + x6 + x7)
        x9 = Conv(x8)
        x10 = Conv (x8 + x9)
        x11 = Conv (x8 + x9 + x10)
        x12 = GAP(x11)
        x13 = FC(x12)
    Uses ReLU and BN wherever applicable
    Uses CIFAR10 as the dataset
    Your target is 75% in less than 40 Epochs
    Once Done:
        Paste the 

        code in QuizDNN.py
        Paste the code of your Colab (or computer's) Notebook
        Paste the complete training log
        Paste the link to your Google Colab Notebook file (or you GitHub Repo)

If there is timing mismatch in any of the files confirming that you attempted the quiz outside the window, then you'll get -50% of this quiz marks. 



