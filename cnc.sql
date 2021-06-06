-- phpMyAdmin SQL Dump
-- version 5.1.0
-- https://www.phpmyadmin.net/
--
-- Anamakine: 127.0.0.1
-- Üretim Zamanı: 17 May 2021, 21:40:53
-- Sunucu sürümü: 10.4.18-MariaDB
-- PHP Sürümü: 8.0.3

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Veritabanı: `graduationproject`
--

-- --------------------------------------------------------

--
-- Tablo için tablo yapısı `cnc`
--

CREATE TABLE `cnc` (
  `cnc_id` int(11) NOT NULL,
  `cnc_registrationNo` text NOT NULL,
  `cnc_machineId` int(11) NOT NULL,
  `cnc_volt` int(11) NOT NULL,
  `cnc_rotate` int(11) NOT NULL,
  `cnc_pressure` int(11) NOT NULL,
  `cnc_vibration` int(11) NOT NULL,
  `cnc_error1count` int(11) NOT NULL,
  `cnc_error2count` int(11) NOT NULL,
  `cnc_error3count` int(11) NOT NULL,
  `cnc_error4count` int(11) NOT NULL,
  `cnc_error5count` int(11) NOT NULL,
  `cnc_model` int(11) NOT NULL,
  `cnc_age` int(11) NOT NULL,
  `cnc_failure` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dökümü yapılmış tablolar için indeksler
--

--
-- Tablo için indeksler `cnc`
--
ALTER TABLE `cnc`
  ADD PRIMARY KEY (`cnc_id`);

--
-- Dökümü yapılmış tablolar için AUTO_INCREMENT değeri
--

--
-- Tablo için AUTO_INCREMENT değeri `cnc`
--
ALTER TABLE `cnc`
  MODIFY `cnc_id` int(11) NOT NULL AUTO_INCREMENT;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
