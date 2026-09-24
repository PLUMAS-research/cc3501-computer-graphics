"""Helpers para configurar Framebuffer Objects (FBO)."""
import pyglet
import pyglet.gl as GL


def create_depth_framebuffer(size, border_color=(1.0, 1.0, 1.0, 1.0)):
    """
    Crea un FBO con solo depth attachment, sin color attachment.

    Devuelve la tupla (framebuffer, depth_texture). El depth attachment
    es una textura cuadrada de size x size, formato GL_DEPTH_COMPONENT32,
    filtrado lineal y wrap CLAMP_TO_BORDER.

    El color de borde (default 1.0) hace que cualquier fragmento que caiga
    fuera del frustum de la luz lea "profundidad máxima" al samplear el
    shadow map, y por lo tanto nunca quede en sombra.

    Sin color attachment hay que avisarle al driver que no se va a escribir
    ni leer color (glDrawBuffer y glReadBuffer en GL_NONE), si no el FBO
    queda incompleto en algunos drivers conformantes.
    """
    depth_texture = pyglet.image.Texture.create(
        size, size,
        internalformat=GL.GL_DEPTH_COMPONENT32,
        fmt=GL.GL_DEPTH_COMPONENT,
        min_filter=GL.GL_LINEAR,
        mag_filter=GL.GL_LINEAR,
    )

    GL.glBindTexture(GL.GL_TEXTURE_2D, depth_texture.id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_BORDER)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_BORDER)
    border = (GL.GLfloat * 4)(*border_color)
    GL.glTexParameterfv(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_BORDER_COLOR, border)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    framebuffer = pyglet.image.Framebuffer()
    framebuffer.attach_texture(depth_texture, attachment=GL.GL_DEPTH_ATTACHMENT)

    framebuffer.bind()
    GL.glDrawBuffer(GL.GL_NONE)
    GL.glReadBuffer(GL.GL_NONE)
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        print(f"[framebuffers] FBO de profundidad incompleto: status={status}")
    framebuffer.unbind()

    return framebuffer, depth_texture


def _create_texture(width, height, internalformat, fmt, filter_mode):
    """Textura vacía del tamaño pedido, sin mipmaps y sin repetición."""
    texture = pyglet.image.Texture.create(
        width, height,
        internalformat=internalformat,
        fmt=fmt,
        min_filter=filter_mode,
        mag_filter=filter_mode,
    )
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture.id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
    return texture


def create_geometry_framebuffer(width, height):
    """
    Crea un FBO con dos adjuntos de color y uno de profundidad.

    Devuelve (framebuffer, position_texture, normal_texture). Las dos texturas
    de color son GL_RGBA16F, porque guardan coordenadas y normales en espacio
    de vista: valores con signo y fuera del rango [0, 1] que un formato de un
    byte por canal no puede representar.

    Escribir en dos adjuntos a la vez (MRT) exige declararlos con glDrawBuffers,
    y el fragment program los elige con `layout(location = ...)`. Sin esa
    llamada, el driver escribe solo en el adjunto 0 y el segundo queda vacío.
    El filtrado es GL_NEAREST: interpolar posiciones de dos superficies
    distintas da un punto que no está en ninguna de las dos.
    """
    position_texture = _create_texture(
        width, height, GL.GL_RGBA16F, GL.GL_RGBA, GL.GL_NEAREST
    )
    normal_texture = _create_texture(
        width, height, GL.GL_RGBA16F, GL.GL_RGBA, GL.GL_NEAREST
    )
    depth_texture = _create_texture(
        width, height, GL.GL_DEPTH_COMPONENT32, GL.GL_DEPTH_COMPONENT, GL.GL_NEAREST
    )

    framebuffer = pyglet.image.Framebuffer()
    framebuffer.attach_texture(position_texture, attachment=GL.GL_COLOR_ATTACHMENT0)
    framebuffer.attach_texture(normal_texture, attachment=GL.GL_COLOR_ATTACHMENT1)
    framebuffer.attach_texture(depth_texture, attachment=GL.GL_DEPTH_ATTACHMENT)

    framebuffer.bind()
    attachments = (GL.GLenum * 2)(GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT1)
    GL.glDrawBuffers(2, attachments)
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        print(f"[framebuffers] FBO de geometría incompleto: status={status}")
    framebuffer.unbind()

    return framebuffer, position_texture, normal_texture


def create_single_channel_framebuffer(width, height):
    """
    Crea un FBO con un adjunto de color de un canal y sin profundidad.

    Devuelve (framebuffer, texture). Sirve para las pasadas de pantalla
    completa que producen un valor por píxel, como el factor de oclusión y su
    desenfoque: no hay geometría que ordenar, así que el buffer de profundidad
    no hace falta.
    """
    texture = _create_texture(width, height, GL.GL_R16F, GL.GL_RED, GL.GL_LINEAR)

    framebuffer = pyglet.image.Framebuffer()
    framebuffer.attach_texture(texture, attachment=GL.GL_COLOR_ATTACHMENT0)

    framebuffer.bind()
    status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
    if status != GL.GL_FRAMEBUFFER_COMPLETE:
        print(f"[framebuffers] FBO de un canal incompleto: status={status}")
    framebuffer.unbind()

    return framebuffer, texture
