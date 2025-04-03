# Applied Machine Learning
---
X-rays are widely used in medical practice. They can be used to identify various diseases. However, a diagnosis depends on a doctor's experience, which can lead to improper treatment. Modern methods of artificial intelligence and pattern recognition make it possible to create expert systems that allow you to establish a diagnosis automatically.

This lab will show you how to upload images, transform them, and determine the basic features that underlie diseases classification.

Two different approaches to the classification of images (diseases) will be shown:
1. Different classical methods and their comparison 
2. Convolutional Neural Networks.
---
## Part 1:Classical Machine Learning Methods for Diagnosis
---
##### 

<h3>Results from Part 1: Classical Machine Learning Methods</h3>
<p>In the <strong>ML in Healthcare</strong> notebook, we will only cover Part 1, focusing on different classical methods. The notebook should produce the following results:</p>

<table>
  <thead>
    <tr>
      <th>Classifier</th>
      <th>Test Accuracy</th>
      <th>Train Accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Logistic Regression</td><td>0.90</td><td>0.85</td></tr>
    <tr><td>Nearest Neighbors</td><td>0.86</td><td>0.70</td></tr>
    <tr><td>Linear SVM</td><td>0.79</td><td>0.71</td></tr>
    <tr><td>RBF SVM</td><td>1.00</td><td>0.48</td></tr>
    <tr><td>Gaussian Process</td><td>0.78</td><td>0.65</td></tr>
    <tr><td>Decision Tree</td><td>0.90</td><td>0.61</td></tr>
    <tr><td>Random Forest</td><td>0.90</td><td>0.61</td></tr>
    <tr><td>Neural Net</td><td>0.93</td><td>0.83</td></tr>
    <tr><td>AdaBoost</td><td>0.85</td><td>0.58</td></tr>
    <tr><td>Naive Bayes</td><td>0.67</td><td>0.59</td></tr>
    <tr><td>QDA</td><td>0.78</td><td>0.80</td></tr>
  </tbody>
</table>

<p><strong>Output as a plot:</strong></p>
<img src="https://raw.githubusercontent.com/asheshghosh/Applied-Machine-Learning/main/Accuracy%20of%20Classifiers.png" alt="Sample Output" width="1000">


---

---
## Part 2:Convolutional Neural Network (CNN) for Diagnosis
---
<p><strong>Output as a plot:</strong></p>
<img src="https://raw.githubusercontent.com/asheshghosh/Applied-Machine-Learning/main/Accuracy%20CNN.png" alt="Sample Output" width="1000">
