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
