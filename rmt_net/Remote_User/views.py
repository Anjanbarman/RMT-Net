
from django.shortcuts import render, redirect, get_object_or_404
import re
import string
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# Create your views here.
from Remote_User.models import ClientRegister_Model,credit_scoring_detection,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Register1(request):

    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Prediction_Breast_Cancer_Detection_Type(request):
    if request.method == "POST":

        if request.method == "POST":
            Customer_Id= request.POST.get('Customer_Id')
            Name= request.POST.get('Name')
            Age= request.POST.get('Age')
            Occupation= request.POST.get('Occupation')
            Annual_Income= request.POST.get('Annual_Income')
            Monthly_Inhand_Salary= request.POST.get('Monthly_Inhand_Salary')
            Num_Bank_Accounts= request.POST.get('Num_Bank_Accounts')
            Num_Credit_Card= request.POST.get('Num_Credit_Card')
            Interest_Rate= request.POST.get('Interest_Rate')
            Num_of_Loan= request.POST.get('Num_of_Loan')
            Type_of_Loan= request.POST.get('Type_of_Loan')
            Delay_from_due_date= request.POST.get('Delay_from_due_date')
            Num_of_Delayed_Payment= request.POST.get('Num_of_Delayed_Payment')
            Changed_Credit_Limit= request.POST.get('Changed_Credit_Limit')
            Num_Credit_Inquiries= request.POST.get('Num_Credit_Inquiries')
            Credit_Mix= request.POST.get('Credit_Mix')
            Outstanding_Debt= request.POST.get('Outstanding_Debt')
            Credit_Utilization_Ratio= request.POST.get('Credit_Utilization_Ratio')
            Credit_History_Age= request.POST.get('Credit_History_Age')
            Payment_of_Min_Amount= request.POST.get('Payment_of_Min_Amount')
            Total_EMI_per_month= request.POST.get('Total_EMI_per_month')
            Amount_invested_monthly= request.POST.get('Amount_invested_monthly')
            Payment_Behaviour= request.POST.get('Payment_Behaviour')
            Monthly_Balance= request.POST.get('Monthly_Balance')


        detection_accuracy.objects.all().delete()

        data = pd.read_csv("Datasets.csv", encoding='latin-1')

        def apply_results(label):
            if (label == 0):
                return 0  # Good
            elif (label == 1):
                return 1  # Poor

        data['Results'] = data['Label'].apply(apply_results)

        x = data['Customer_Id'].apply(str)
        y = data['Results']

        cv = CountVectorizer(lowercase=False, strip_accents='unicode', ngram_range=(1, 1))

        x = cv.fit_transform(x)
        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("SGD Classifier")
        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier(loss='hinge', penalty='l2', random_state=0)
        sgd_clf.fit(X_train, y_train)
        sgdpredict = sgd_clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, sgdpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, sgdpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, sgdpredict))
        models.append(('SGDClassifier', sgd_clf))


        print("Gradient Boosting Classifier")

        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
            X_train,
            y_train)
        clfpredict = clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, clfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, clfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, clfpredict))
        models.append(('GradientBoostingClassifier', clf))

        # SVM Model
        print("SVM")
        from sklearn import svm
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Random Forest Classifier")
        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train, y_train)
        rfpredict = rf_clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, rfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, rfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, rfpredict))
        models.append(('RandomForestClassifier', rf_clf))



        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        Customer_Id1 = [Customer_Id]
        vector1 = cv.transform(Customer_Id1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = pred.replace("]", "")

        prediction = int(pred1)

        if prediction == 0:
            val = 'Good'
        elif prediction == 1:
            val = 'Poor'


        print(prediction)
        print(val)

        credit_scoring_detection.objects.create(
Customer_Id=Customer_Id,
Name=Name,
Age=Age,
Occupation=Occupation,
Annual_Income=Annual_Income,
Monthly_Inhand_Salary=Monthly_Inhand_Salary,
Num_Bank_Accounts=Num_Bank_Accounts,
Num_Credit_Card=Num_Credit_Card,
Interest_Rate=Interest_Rate,
Num_of_Loan=Num_of_Loan,
Type_of_Loan=Type_of_Loan,
Delay_from_due_date=Delay_from_due_date,
Num_of_Delayed_Payment=Num_of_Delayed_Payment,
Changed_Credit_Limit=Changed_Credit_Limit,
Num_Credit_Inquiries=Num_Credit_Inquiries,
Credit_Mix=Credit_Mix,
Outstanding_Debt=Outstanding_Debt,
Credit_Utilization_Ratio=Credit_Utilization_Ratio,
Credit_History_Age=Credit_History_Age,
Payment_of_Min_Amount=Payment_of_Min_Amount,
Total_EMI_per_month=Total_EMI_per_month,
Amount_invested_monthly=Amount_invested_monthly,
Payment_Behaviour=Payment_Behaviour,
Monthly_Balance=Monthly_Balance,
Prediction=val)

        return render(request, 'RUser/Prediction_Breast_Cancer_Detection_Type.html',{'objs': val})
    return render(request, 'RUser/Prediction_Breast_Cancer_Detection_Type.html')
