## Week 1

* **ML**: que la computadora mejore su performance P en la tarea T a partir de la experiencia E.
* **Classification problem**: valor de salida discreto
* **Reggresion**: predice valores de salida continuos.
* **Supervised**: reggresion o classification. Se conoce un conjunto de respuestas correctas de antemano, y luego se quiere producir una respuesta para un dato nuevo.
* **Unsupervised**: sin etiquetar? no hay "respuesta correcta". Se obtienen ciertos resultados como agrupación, separación, y otros.
* **Notación**:
	* $m$: # training examples
	* $x's$: input o feature
	* $y's$: output o target
	* $(x^{(i)},y^{(i)})$: obvio
* Training set ---> Learning algorithm ---> hypothesis h (aunque no tenga sentido que se llame así). $h : X \rightarrow Y$
* **Linear regression**:
	* h es función lineal
	* $h_{\theta}(x) = \theta_0 + \theta_1 x$
	* Lo que tenemos que hacer es hallar $\theta_0$ y $\theta_1$, de forma que la predicción de h sea lo más correcta posible, basándonos en el training set.
	* Para eso vamos a minimizar la función $J(\theta_0, \theta_1)$, que es un promedio de la suma de los cuadrados de las diferencias entre $h(x^{(i)})$ y $y^{(i)}$, para todos los $i$.
	* Llamamos a $J$ **cost function**.
* *Intuition I*: Simplified
	* Tomo $\theta_0 = 0$
	* Entonces queda $J(\theta_1)$, que es un promedio de la suma de los cuadrados de las diferencias entre $\theta_1 x^{(i)}$ y $y^{(i)}$, para todos los $i$.
* *Intuition II*: nos queda una parábola pero en 3D, por lo que tenemos elipses con el mismo J.
* *Gradient descent*: moverse buscando el mínimo, con cierta automatización
* *Gradient descent intuition*: las derivadas parciales nos dicen para dónde movernos, y el *rate* determina una parte del cuánto.
* En linear regression, la cost function siempre es convex, así que todo liso.
* Se puede calcular el vector de predicciones multiplicando la matriz de entrada por el vector de parámetros ($\theta_0$ y $\theta_1$).
* Propiedades de las matrices:
	* *Asociativas*: $A \times (B \times C) = (A \times B) \times C$
	* *NO conmutativas*: $A \times B = B \times A$
	* *Identidad*

## Week 2

### Multivariable linear regression

Ahora la función de predicción es $f : \mathbb{R}^{n+1}$, siendo $n$ la cantidad de variables de entrada. El vector de entrada tiene dimensión $n+1$ porque se agrega una constante $x_0=1$ que acompaña al término suelto de la ecuación lineal. Ahora:

$$f(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_n x_n = \theta^Tx$$

Nota: $x^{(i)}_j$ denota al dato correspondiente a la columna $j$ de la fila $i$ del *training set*.

La función costo es similar solo que para $n$.

##### Optimización de gradient descent

* **Feature scaling**:
	* Si sabemos que las variables ranguean entre $a$ y $b$, dividimos $x_i$ por $b - a$, lo que nos da $0 \le x_i \le 1$.
	* La idea sería alcanzar algo cercano a $-1 \le x_i \le 1$.
* **Mean normalization**:
	* Variables rangueando entre $a$ y $b$, con promedio $m$: $(x_i - m_i) / (b_i - a_i)$.
	* La idea es alcanzar algo como $-0,5 \le x_i \le 0,5$.
	* Aparentemente, a $(b_i - a_i)$ se le dice rango también. Y desviación estándar parece que también. O quizá la desviación estándar es otro número que podemos usar allí.

##### Elección del $\alpha$

* Si es muy grande va a fracasar la minimización (no converge o es muy lenta)
* Si es muy chico va a ser muy lenta
* Qué probar: $0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1$

¿Cuándo convergió? Cuando decrece con tasa menor a un cierto número definido.

##### Polynomial regression

Por ejemplo, en el ejemplo donde $x_1$ es $size$, cuyo rango es $[1,1000]$, agregamos una variable nueva que sea $\sqrt{size}$. Entonces la función nos queda 
$$f(x) = \theta_0 \times 1 + \theta_1 \times size + \theta_2 \times \sqrt{size}$$
$$f(x) = \theta_0 x_0 + \theta_1 x_1 + \theta_2 x_2$$
Luego, al aplicar *feature scaling*, $x_2$ *es* $\sqrt{size}$, por lo tanto su rango es $[1,\sqrt{1000}]$, y quedaría $$x_2 / \sqrt{1000}$$

### Normal equation

Minimización analítica de la función costo, hallando el $\theta$ correcto.

$$\theta = (X^TX)^{-1}X^Ty$$

Es más fácil que GD, pero no escala (es $O(n^3)$).

* $X$ es la matriz construída a partir de los datos de entrada del training set, agregando $x_0$. Tiene dimensión $m \times n+1$.
* $y$ es el vector de valores de salida del training set. Tiene dimensión $m$.

Si $(X^TX)$ fuera singular (sin inversa), la función `pinv` de Octave igualmente calcula el $theta$ que minimiza $J$.

### Octave

##### Comandos

* `while <cond>, <comandos>; end;`
* `for i=1:10, <comandos>; end;`
* `for i=ind, <comandos>; end;` con `ind=1:10`
* `if <cond>, <comandos>; [else|elseif <cond>,] <comandos>; end;`

##### Funciones

```
function y = double(x)
y = 2*x

function [a,b] = doubleAndHalf(x)
a = double(x)
b = x/2

function J = costFunctionJ(X, y, theta)

m = size(X,1);
predictions = X*theta;
sqrErr = (predictions - y).^2;

J = 1/(2*m) * sum(sqrErr);
```

##### Vectorización

En vez de usar un `for` para la sumatoria de los elementos de dos vectores, aplicamos la operación matemática que calcula lo que queremos.

Por ejemplo, si queremos calcular la multiplicación de dos vectores $X$ y $\theta$ elemento a elemento, hacemos $X^T\theta$.

Si queremos calcular un paso de *gradient descent*, en vez de un `for` que calcule $\theta_i - \alpha 1/m \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) x_0^{(i)}$ para `i = 0:2`, armamos unos buenos vectores:

* $\theta$ tiene los coeficientes de la función de predicción.
* $\alpha$ es la aceleración de cambio.
* $\delta = 1/m \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) x_0^{(i)}$

## Week 3

Arrancamos con *classification*, donde los valores de salida son discretos. Vimos que no sirve usar linear regression, lo que tiene sentido porque una función que siempre crece no puede mantenerse dentro de un conjunto finito de valores de salida. Se aplica entonces **logistic regression model**.

### Logistic regression model

Ahora la función de predicción es $h_\theta(x) = g(\theta^Tx)$, con $g(z) = \frac{1}{1+e^{-z}}$. $h$ cumple la condición $0 \le h_\theta(x) \le 1$. La función $g$ mantiene a todos los valores dentro del intervalo $(0,1)$. El resultado de $h$ es entonces la probabilidad de que la salida sea $1$. Si $\theta^Tx \ge 0$, entonces $h(\theta^Tx) \ge 0.5$ y por lo tanto $y=1$.

### Decision boundary

$\theta^Tx$ representa a una función polinómica que representará un límite entre los datos de entrada: si están por encima (o por debajo) de ese límite, el resultado será positivo, caso contrario negativo.

### Cost function

$J(\theta) = \frac{1}{m} \sum_{i=1}^m Cost(h_\theta(x^{(i)}),y^{(i)})$

$Cost(h_\theta(x),y) = -\log(h_\theta(x))$ si $y = 1$

$Cost(h_\theta(x),y) = -\log(1 - h_\theta(x))$ si $y = 0$

Simplified $Cost$:

$Cost(h_\theta(x),y) = -y \log(h_\theta(x)) -(1-y) \log(1 - h_\theta(x))$

Final $J$:

$J(\theta) = -\frac{1}{m} \sum_{i=1}^m y \log(h_\theta(x)) + (1-y) \log(1 - h_\theta(x))$

### Gradient descent

Igual...?

### Overfitting

Sarparse agregando parámetros a $\theta$, logrando que $h$ se ajuste *muy* bien a los valores del training set, pero no prediga bien nuevas entradas.

##### Solución: regularización

Primero cambiamos la función costo: agregamos un término que penalice el tamaño de los parámetros en $\theta$: $\lambda \sum_{j=1}^n \theta_j^2$.

Luego, en *gradient descent*, cambiamos el cálculo de cada $\theta_j$: $\theta_j (1 - \alpha \frac{\lambda}{m} - ...$.

## Week 4

Las redes neuronales son composiciones de funciones de logistic regression.

Llamamos *neurona* a la función $h_{\theta}(x) = \frac{1}{1+e^{-\theta^Tx}}$, con entrada $x = [x_0; x_1; x_2; x_3]$ y parámetros $\theta = [\theta_0; \theta_1; \theta_2; \theta_3]$ (también llamados *weights*).

Las redes neuronales son grupos de neuronas organizadas en capas, donde:

* la entrada de cada neurona de la capa $k+1$ será el vector de las salidas de cada neurona de la capa $k$;
* los parámetros de cada neurona se definirán en una matriz $\Theta^{(j)}$ ($\Theta$ es una lista de matrices entonces) donde la fila $1$ corresponde a los parámetros de la neurona $1$ de la capa (el tamaño de la matriz de cada capa será, entonces: *neuronas* $\times$ *entradas*).

Se puede ver a cada elemento de cada capa como una unidad de activación. Serán entonces unidades de activación las neuronas y las entradas.

Una forma de computar el resultado de la red ante una entrada, es calcular $a^{(2)}$ como $g(\Theta^{(1)}x)$, luego $a^{(3)}$ como $g(\Theta^{(2)}a^{(2)})$, y en forma genérica $a^{(i)}$ como $g(\Theta^{(i-1)}a^{(i-1)})$. A esto se le llama *forward propagation*.

`XNOR`: la red neuronal le permitió, al tener diferentes neuronas con su propio significado, combinar significados y obtener el significado buscado. Podemos entonces ver efectivamente a cada neurona como una función que calcula un cierto resultado para la entrada, que luego podemos utilizar de la forma que nos convenga. Un ejemplo más interesante es el de los números escritos a mano, donde se puede visualizar que cada neurona genera una imagen modificada basada en la unidad anterior, y cómo se complejizan las capa al avanzar en "profundidad".