from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address = models.CharField(max_length=3000)
    gender = models.CharField(max_length=300)

class credit_scoring_detection(models.Model):

    Customer_Id= models.CharField(max_length=300)
    Name= models.CharField(max_length=300)
    Age= models.CharField(max_length=300)
    Occupation= models.CharField(max_length=300)
    Annual_Income= models.CharField(max_length=300)
    Monthly_Inhand_Salary= models.CharField(max_length=300)
    Num_Bank_Accounts= models.CharField(max_length=300)
    Num_Credit_Card= models.CharField(max_length=300)
    Interest_Rate= models.CharField(max_length=300)
    Num_of_Loan= models.CharField(max_length=300)
    Type_of_Loan= models.CharField(max_length=300)
    Delay_from_due_date= models.CharField(max_length=300)
    Num_of_Delayed_Payment= models.CharField(max_length=300)
    Changed_Credit_Limit= models.CharField(max_length=300)
    Num_Credit_Inquiries= models.CharField(max_length=300)
    Credit_Mix= models.CharField(max_length=300)
    Outstanding_Debt= models.CharField(max_length=300)
    Credit_Utilization_Ratio= models.CharField(max_length=300)
    Credit_History_Age= models.CharField(max_length=300)
    Payment_of_Min_Amount= models.CharField(max_length=300)
    Total_EMI_per_month= models.CharField(max_length=300)
    Amount_invested_monthly= models.CharField(max_length=300)
    Payment_Behaviour= models.CharField(max_length=300)
    Monthly_Balance= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



