---
layout: post
title:  "Entrenando un detector de objetos en Tensorflow"
date:   2019-03-14 10:37:00
categories: blog
---
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Disclaimer</h1>
<h3 style="font-size: 30px; font-family: Verdana;text-align: center;">Cuando NO intentar reproducir este proyecto:</h3>
<ul>
<li>Si tienes un l&iacute;mite de tiempo que no deja espacio para "oopsies".</li>
<li>Si usas windows y estas convencido de que de todas maneras deber&iacute;a funcionar.</li>
<li>Si usas windows y deseas conservar tu sanidad mental.</li>
<li>Si usas windows. Si usas macOS Mojave y no estas listo para una aventura.</li>
</ul>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Introducci&oacute;n</h1>
<p>Este blog fue realizado para la clase de Redes Neuronales impartida en el semestre 2019-1 por el profesor Julio Waissman Vilanova en la Universidad de Sonora. El proyecto consist&iacute;a en utilizar una red convolucional para resolver un problema en concreto. Yo eleg&iacute; la detecci&oacute;n de objetos y reducci&oacute;n del modelo para uso en dispositivos m&oacute;viles.</p>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Arquitecturas utilizadas</h1>
<h3 style="font-size: 30px; font-family: Verdana;text-align: center;">Faster RCNN</h3>
<h3 style="font-size: 30px; font-family: Verdana;text-align: center;">SSD Mobilenet</h3>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Comenzar el entrenamiento</h1>
<h3 style="font-size: 30px; font-family: Verdana;text-align: center;">Conseguir datos</h3>
<div class="textarea">
<div>El plan original era hacer un detector de insectos con especies end&eacute;micas, para evitar utilizar categor&iacute;as que el modelo&nbsp;<span class="googie_link">pre-entrenado</span>&nbsp;ya conoc&iacute;a. Sin embargo, debido a mi nulo acceso a insectos en un ambiente en seguro (y miedo), hice un peque&ntilde;o cambio: tom&eacute; tres figuras de&nbsp;<span class="googie_link">quasi-insectos</span>.&nbsp;</div>
<div><br />Si&nbsp;<span class="googie_link">google</span>&nbsp;tiene suficientes im&aacute;genes de lo que quieres detectar, hay muchas extensiones que descargan todas las im&aacute;genes de una p&aacute;gina web a un .<span class="googie_link">zip</span>. Si no, va a tomar un poco m&aacute;s de tiempo pero puedes tomar las fotos tu mismo. No importa si tu celular/tableta tiene mala c&aacute;mara, de igual manera hay que modificar su tama&ntilde;o.</div>
<div><br />Te recomiendo que tengas una saludable cantidad de im&aacute;genes donde se vea claramente el objeto que quieres detectar, as&iacute; como im&aacute;genes donde haya&nbsp;<span class="googie_link">distractores</span>&nbsp;en el fondo. Repite para todas tus clases.</div>
<div>&nbsp;</div>
<div>En mi caso entren&eacute; para que reconociera tres clases de quasi-insectos: chimera (de el juego resistance para PS3), randall (de la pel&iacute;cula Monsters Inc.) y demogorgon (de la serie de Netflix Stranger Things).</div>
</div>
<h3 style="font-size: 30px; font-family: Verdana;text-align: center;">Preparar los datos</h3>
<p>En uno de los miles tutoriales que segu&iacute; (todos los links estar&aacute;n al final del post) proporcionaba un script sencillo para hacer un resize de las imagenes. Lo encontrar&aacute;s en el proyecto de github.</p>
<h3 style="font-size: 30px; font-family: Verdana;text-align: center;">Preparar las etiquetas</h3>
<p>La parte m&aacute;s divertida (no) de todo el proyecto, dibujar los recuadros en absolutamente todas las imagenes que utilizaras. Para eso, lo m&aacute;s facil es descargar&nbsp;<a href="https://github.com/tzutalin/labelImg">Labellmg.</a> Es muy intuitivo, b&aacute;sicamente arrastras el "bouding box" y lo clasificas para todas las clases que aparezcan en la imagen.</p>
<h3 style="font-size: 30px; font-family: Verdana;text-align: center;">Crear el archivo CSV</h3>
<p>Este paso crea un archivo CSV que contiene la informaci&oacute;n de las imagenes(tama&ntilde;o, localizaci&oacute;n del "bounding box" etc.).&nbsp;</p>
<h3 style="font-size: 30px; font-family: Verdana;text-align: center;">Entrenar</h3>
<p>Para entrenar tenemos que descargar un modelo de&nbsp;<a href="https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md">aqu&iacute;</a>. Yo utilic&eacute; inicialmente el modelo rfcn_resnet101_coco. Lo configur&eacute; y entrene, con muy buenos resultados, para darme cuenta de que la arquitectura faster-rcnn no es compatible con el convertidor a tflite, que es el tipo de modelo que se puede utilizar en dispositivos m&oacute;viles. No lo v&iacute; en ning&uacute;n lado (hasta que me dio el error y busque eso espec&iacute;ficamente), puede ser que no buscara correctamente pero pens&eacute; que ser&iacute;a bueno mencionarlo.&nbsp;</p>
<p>Despu&eacute;s de mi fracaso, volv&iacute; a comenzar el proceso con el modelo ssd_mobilenet_v1_coco, con muy buenos resultados y mucho mejores tiempos.</p>
<p>Si mi disclaimer no te detuvo de intentar hacer este proyecto en windows, perm&iacute;teme contarte la historia de un modelito que no quer&iacute;a ser entrenado; error tras error, reinstalando Tensorflow de todas las maneras imaginables (siguiendo consejos de internet), probando con multiples combinaciones de versiones de Tensorflow-Cuda-etc, las cuales todas se supone deber&iacute;an funcionar. Nada funciono. &iquest;La soluci&oacute;n? Cambiarme a una Mac. Haciendo exactamente lo mismo que hice en windows funcion&oacute; a la primera.</p>
<p>Me parece que la mejor opci&oacute;n seg&uacute;n mi basta experiencia en el tema (google) es Linux, pero Mac es buen segundo lugar.</p>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Probar los resultados</h1>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Convertir a tflite</h1>
<h1 style="font-size: 40px; font-family: Verdana; text-align: center;">Pasar a Android app</h1>
<p><img src="../../assets/img/app1.jpg" alt="mimers" /></p>
