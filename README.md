<!-- init organization banner -->
<p align="center">
    <img src="https://www.aaaimx.org/img/other/aaaimx-ist.png" width="400" alt="AAAIMX"><br><br>
    <a href="https://www.aaaimx.org/" target="_blank">
        <img src="https://img.shields.io/badge/website-AAAI%20Student%20Chapter%20M%C3%A9xico-yellow">
    </a>
    <a href="https://web.facebook.com/aaaimx/" target="_blank">
        <img src="https://img.shields.io/badge/follow%20us-%40aaaimx-blue">
    </a>
    <a href="https://www.paypal.me/aaaimx" target="_blank">
        <img src="https://img.shields.io/badge/donate-support%20us-green">
    </a>
</p>
<!-- end banner -->

# Redes Neuronales con TensorFlow

Acercamiento a las redes neuronales con la librería de Python TensorFlow.
Se estudiará como funciona una red neuronal, como entrenarlas, programarlas, pasarlas a otro lenguaje de programación, etc. hasta manejar redes complejas en TensorFlow. Como clasificadores de imágenes.

## Requisitos
- Python 3.7
- [Entorno virtual](https://edgardorl.com/blog/instalar-python-pip-y-virtualenv-en-windows-10/) listo para usar (opcional)
- Editor de código [VSCode](https://code.visualstudio.com/download) (Recomendado)
- Tener conocimientos sobre:
  * Uso de *for*, *while*, *if*, *else*; _tipos de datos_, _arreglos_(se manejaran hasta de 3 dimensiones).
  * Derivadas parciales
  * Multiplicación y suma de matrices
  
También pueden trabajar con Google Colaboratory.

## Instalación

**$**: Indica que es un comando de terminal

``` bash
$ pip install virtualenv # opcional
$ virtualenv venv # opcional
$ .\venv\Scripts\activate # opcional
$ pip install -r .\requirements.txt # necesario
$ deactivate # salir
```
## Enlaces útiles
[Instalar python, pip y virtualenv en Windows](https://edgardorl.com/blog/instalar-python-pip-y-virtualenv-en-windows-10/)

[Descargar VSCode](https://code.visualstudio.com/download)

## Troubleshooting
> **NOTA:** La libreria Tensorflow no funciona en Python 3.8 o superiores, procure instalar una version de python 3.7 o menor.

> `$ python get-pip.py` - Solución al error: "El término 'pip' no se reconoce como nombre de un cmdlet, 
función, archivo de script o programa ejecutable." o descarga el script [get-pip.py](https://bootstrap.pypa.io/get-pip.py)



> Solución al error de PowerShell “No se puede cargar el archivo porque la ejecución de scripts está deshabilidada en este sistema”
[Error de activación de entorno virtual](https://protegermipc.net/2018/11/22/permitir-la-ejecucion-de-scripts-powershell-en-windows-10/)
