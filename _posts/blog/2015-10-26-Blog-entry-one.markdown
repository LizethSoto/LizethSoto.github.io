---
layout: post
title:  "Entrenando un detector de objetos en Tensorflow"
date:   2019-03-14 10:37:00
categories: blog
---
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Disclaimer</h1>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">Cuando NO intentar reproducir este proyecto:</h1>
<ul>
<li>Si no eres un hacker nivel 4 o en su defecto tienes muy poca paciencia</li>
<li>Si tienes un l&iacute;mite de tiempo que no deja espacio para "oopsies".</li>
<li>Si usas macOS Mojave y no estas listo para una aventura.</li>
<li>Si usas windows y estas convencido de que de todas maneras deber&iacute;a funcionar.</li>
<li>Si usas windows y deseas conservar tu sanidad mental.</li>
<li>Si usas windows.</li>
</ul>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Introducci&oacute;n</h1>
<p>Este blog fue realizado para la clase de Redes Neuronales impartida en el semestre 2019-1 por el profesor Julio Waissman Vilanova en la Universidad de Sonora. El proyecto consist&iacute;a en utilizar una red convolucional para resolver un problema en concreto. Yo eleg&iacute; la detecci&oacute;n de objetos y reducci&oacute;n del modelo para uso en dispositivos m&oacute;viles.</p>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Arquitectura utilizada</h1>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">SSD Mobilenet</h1>
<p>Una red convolucional consiste en los siguientes tipos de capas.</p>
<p>INPUT: mantendr&aacute; los valores de p&iacute;xeles sin procesar de la imagen, en este caso una imagen con tres canales de color R, G, B.<br />CONV: Calcular&aacute; la salida de las neuronas que est&aacute;n conectadas a las regiones locales en la entrada, cada una de las cuales calcula un producto de puntos entre sus pesos y una peque&ntilde;a regi&oacute;n a la que est&aacute;n conectadas en el volumen de entrada. <br />RELU: Elimina los elementos negativos, dejando solo los positivos. De esta manera los cambios de color ser&aacute;n menos graduales.<br />POOL: realizar&aacute; una operaci&oacute;n de reducci&oacute;n de resoluci&oacute;n a lo largo de las dimensiones espaciales.<br />FC:Calcular&aacute; los puntajes de cada clase.</p>
<p>Tomando eso en cuenta, la arquitectura de la SSD Mobilenet es la siguiente:</p>
<p><img src="/assets/img/red.png" alt="red" /></p>


<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Comenzar el entrenamiento</h1>
<p>Lo primero que debes hacer es descargar&nbsp;<a href="https://github.com/tensorflow/models/tree/master/research/object_detection">esto</a> del github de Tensorflow. Este proyecto fue realizado en mayo de 2019 y constantemente se le hacen cambios. Es muy probable que dentro de no mucho tiempo ya no sirva, me he topado con ese problema para diferentes cosas haciendo este proyecto. Unas partes necesitan una versi&oacute;n, otras las "rompieron" con la actualizaci&oacute;n etc.&nbsp;</p>
<p>&nbsp;</p>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">Conseguir datos</h1>
<div class="textarea">
<div>El plan original era hacer un detector de insectos con especies end&eacute;micas, para evitar utilizar categor&iacute;as que el modelo&nbsp;<span class="googie_link">pre-entrenado</span>&nbsp;ya conoc&iacute;a. Sin embargo, debido a mi nulo acceso a insectos en un ambiente en seguro (y miedo), hice un peque&ntilde;o cambio: tom&eacute; tres figuras de&nbsp;<span class="googie_link">quasi-insectos</span>.&nbsp;</div>
<div><br />Si&nbsp;<span class="googie_link">google</span>&nbsp;tiene suficientes im&aacute;genes de lo que quieres detectar, hay muchas extensiones que descargan todas las im&aacute;genes de una p&aacute;gina web a un .<span class="googie_link">zip</span>. Si no, va a tomar un poco m&aacute;s de tiempo pero puedes tomar las fotos tu mismo. No importa si tu celular/tableta tiene mala c&aacute;mara, de igual manera hay que modificar su tama&ntilde;o.</div>
<div><br />Te recomiendo que tengas una saludable cantidad de im&aacute;genes donde se vea claramente el objeto que quieres detectar, as&iacute; como im&aacute;genes donde haya&nbsp;<span class="googie_link">distractores</span>&nbsp;en el fondo. Repite para todas tus clases.</div>
<div>&nbsp;</div>
<div>En mi caso entren&eacute; para que reconociera tres clases de quasi-insectos: chimera (de el juego resistance para PS3), randall (de la pel&iacute;cula Monsters Inc.) y demogorgon (de la serie de Netflix Stranger Things).</div>
<div>&nbsp;</div>
<div>Les comparto una imagen que captura el momento exacto donde se hizo un atentado contra mi proyecto.</div>
</div>
<p><img src="../../assets/img/DSC_0554.JPG" alt="chamuk1" /></p>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">&nbsp;</h1>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">Preparar los datos</h1>
<p>En uno de los miles tutoriales que segu&iacute; (todos los links estar&aacute;n al final del post) proporcionaba un script sencillo para hacer un resize de las imagenes. Lo encontrar&aacute;s en el proyecto de github.</p>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">&nbsp;</h1>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">Preparar las etiquetas</h1>
<p>La parte m&aacute;s divertida (no) de todo el proyecto, dibujar los recuadros en absolutamente todas las imagenes que utilizaras. Para eso, lo m&aacute;s facil es descargar&nbsp;<a href="https://github.com/tzutalin/labelImg">Labellmg.</a> Es muy intuitivo, b&aacute;sicamente arrastras el "bouding box" y lo clasificas para todas las clases que aparezcan en la imagen.</p>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">&nbsp;</h1>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">Crear el archivo CSV</h1>
<p>Este paso crea un archivo CSV que contiene la informaci&oacute;n de las imagenes(tama&ntilde;o, localizaci&oacute;n del "bounding box" etc.).&nbsp;</p>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">&nbsp;</h1>
<h1 style="font-size: 30px; font-family: Verdana; text-align: center;">Entrenar</h1>
<p>Para entrenar tenemos que descargar un modelo de&nbsp;<a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">aqu&iacute;</a>. Yo utilic&eacute; inicialmente el modelo rfcn_resnet101_coco. Lo configur&eacute; y entrene, con muy buenos resultados, para darme cuenta de que la arquitectura faster-rcnn no es compatible con el convertidor a tflite, que es el tipo de modelo que se puede utilizar en dispositivos m&oacute;viles. No lo v&iacute; en ning&uacute;n lado (hasta que me dio el error y busque eso espec&iacute;ficamente), puede ser que no buscara correctamente pero pens&eacute; que ser&iacute;a bueno mencionarlo.&nbsp;</p>
<p>Despu&eacute;s de mi fracaso, volv&iacute; a comenzar el proceso con el modelo ssd_mobilenet_v1_coco, con muy buenos resultados y mucho mejores tiempos.</p>
<p>Si mi disclaimer no te detuvo de intentar hacer este proyecto en windows, perm&iacute;teme contarte la historia de un modelito que no quer&iacute;a ser entrenado; error tras error, reinstalando Tensorflow de todas las maneras imaginables (siguiendo consejos de internet), probando con multiples combinaciones de versiones de Tensorflow-Cuda-etc, las cuales todas se supone deber&iacute;an funcionar. Nada funciono. &iquest;La soluci&oacute;n? Cambiarme a una Mac. Haciendo exactamente lo mismo que hice en windows funcion&oacute; a la primera.</p>
<p>Me parece que la mejor opci&oacute;n seg&uacute;n mi basta experiencia en el tema (google) es Linux, pero Mac es buen segundo lugar.</p>
<p>Fuera de eso, el entrenamiento es muy "straight forward". Te recomiendo seguir&nbsp;<a href="https://medium.com/datadriveninvestor/training-object-detection-for-windows-with-tensorflow-505ae7d19516">este</a>&nbsp;(desde que fue creado cambiaron la estructura de las carpetas, pero a&uacute;n puedes encontrar el archivo "train.py"&nbsp; a una carpeta llamada "legacy") tutorial para la parte del entrenamiento hasta donde utiliza el archivo "export_inference_graph.py"</p>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">&nbsp;</h1>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Probar los resultados</h1>
<p>Para probar los resultados puedes utilizar dos archivos muy &uacute;tiles sacados de&nbsp;<a href="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10">este</a>&nbsp;tutorial llamados "Object_detection_image.py" y "Object_detection_video.py". S&oacute;lamente tienes que modificar el path a tu modelo congelado, tu labelmap y tu imagen en el caso del primer archivo. Estos son mis resultados.</p>
<p><img src="../../assets/img/cap1.png" alt="cap1" /></p>
<p><img src="../../assets/img/cap2.png" alt="cap2" /></p>
<p><img src="../../assets/img/cap3.png" alt="cap3" /></p>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">&nbsp;</h1>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Convertir a tflite</h1>
<p>B&aacute;sicamente hasta aqui lleg&oacute; mi avance. Lo que segu&iacute;a era exportar el modelo como tflite sdd graph, con un archivo del mismo nombre, lo cual es bastante sencillo. La verdadera dificultad es transformar el modelo a tflite. Originalmente se utilizaba "TOCO", sin embargo esto ya esta descontinuado, de acuerdo con la p&aacute;gina oficial de tensorflow. Ahora lo que se debe utilizar es "TFliteConverter", sin embargo no hubo nada que yo pudiera hacer para lograr que funcionara.</p>
<p>Para poder utilizar el m&eacute;todo, hab&iacute;a que "buildearlo" con bazel, pero tanto bazel como Tensorflow como practicamente todo lo que utilic&eacute; durante el proyecto no funcionaba con las actualizaciones. Bazel me pedia una version, Tensorflow me ped&iacute;a otra, y ambas no eran compatibles.</p>
<p>Hay muy poca informaci&oacute;n al respecto, muchos reportes del problema se cierran por inactividad, e incluso en algunos mencionaban que la gente de algunos paquetes estaban baneando a los usuarios que se quejaban de los bugs.&nbsp;</p>
<p>Mi sospecha es que con la pronta llegada de Tensorflow 2.0, muchos paquetes estan queriendo prepararse con anticipaci&oacute;n y estan dejando de lado el "ahora".&nbsp;</p>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Pasar a Android app</h1>
<p>Cuando por fin logre utilizar, en uno de mis multiples intentos, el TFliteConverter, al pasar mi modelo al demo de detecci&oacute;n de objetos de tensorflow para Android, crashea al instante.&nbsp;</p>
<p>Mi modelo funciona antes de pasarlo por el convertidor, y la aplicaci&oacute;n funciona antes de cambiar el modelo por el mio. Mi sospecha es que a pesar de obtener un archivo.tflite, no se crea de la manera correcta causando el crash.</p>
<p>Es un tema que la verdad me llama mucho la atenci&oacute;n y seguir&eacute; intentando hacer que funcione. Eventualmente lo har&aacute;, ya sea porque descubr&iacute; el problema o porque pas&oacute; tanto tiempo que ya funcionan en armon&iacute;a todas las versiones de nuevo.</p>
<p><img src="../../assets/img/app1.jpg" alt="mimers" /></p>
<p>&nbsp;</p>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Links</h1>
<ul>
<li><a href="https://medium.com/datadriveninvestor/training-object-detection-for-windows-with-tensorflow-505ae7d19516">https://medium.com/datadriveninvestor/training-object-detection-for-windows-with-tensorflow-505ae7d19516</a></li>
<li><a href="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10">https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10</a></li>
<li><a href="https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193">https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193</a></li>
<li><a href="https://towardsdatascience.com/detecting-pikachu-on-android-using-tensorflow-object-detection-15464c7a60cd">https://towardsdatascience.com/detecting-pikachu-on-android-using-tensorflow-object-detection-15464c7a60cd</a></li>
</ul>
