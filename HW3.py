
import numpy
import tools


if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Шаг по пространству
    dx = 3e-3

    # Размер области моделирования
    x = 1.5

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = 3e8

    # Шаг по времени
    dt = Sc * dx / c
    print(f"Шаг временной сетки: {dt} с")

    # Время расчета в отсчетах
    maxTime = 700

    # Размер области моделирования в отсчетах
    maxSize = int(x / dx)

    # Положение источника в отсчетах
    sourcePos = 50

    # Датчики для регистрации поля
    probesPos = [75]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize)

    for probe in probes:
        probe.addData(Ez, Hy)

    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    # Параметры гауссова импульса
    A0 = 100  # ослабление в 0 момент времени по отношение к максимуму
    Am = 100  # ослабление на частоте Fm
    Fm = 1.5e9
    wg = numpy.sqrt(numpy.log(Am)) / (numpy.pi * Fm)
    NWg = wg / dt
    dg = wg * numpy.sqrt(numpy.log(A0))
    NDg = dg / dt

    for t in range(1, maxTime):
        # Граничные условия для поля H
        Hy[-1] = Hy[-2]

        # Расчет компоненты поля H
        Ez_shift = Ez[1:]
        Hy[:-1] = Hy[:-1] + (Ez_shift - Ez[:-1]) * Sc / W0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Hy[sourcePos - 1] -= (Sc / W0) * \
            numpy.exp(-((t - NDg - sourcePos) / NWg) ** 2)
        # Hy[sourcePos - 1] -= (Sc / W0) * numpy.exp(-(t - 30.0) ** 2 / 100.0)

        # Граничные условия для поля E
        Ez[0] = Ez[1]

        # Расчет компоненты поля E
        Hy_shift = Hy[:-1]
        Ez[1:] = Ez[1:] + (Hy[1:] - Hy_shift) * Sc * W0

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos] += Sc * \
            numpy.exp(-(((t + 0.5) - (sourcePos - 0.5) - NDg) / NWg) ** 2)
        # Ez[sourcePos] += Sc * numpy.exp(-((t + 0.5) - (-0.5) - 30.0) ** 2 / 100.0)

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if t % 50 == 0:
            display.updateData(display_field, t)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1)

    # Получение спектра сигнала в датчике
    F = tools.Furie(1024, probe.E, dt)
    F.FFT()
